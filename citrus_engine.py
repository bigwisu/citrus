"""
CITRUS ENGINE (v1.0)
Core logic for: From Static Keywords to Adaptive Vectors
Author: Wisudanto C. Suntoyo
License: MIT

This module handles:
1. Dual-Mode Vectorization (IBM Granite & Gemini)
2. Vector Database Operations (sqlite-vec)
3. Mathematical Calibration (HiLAT Knee Detection)
4. Jaccard Orthogonality Checks
"""
# citrus_engine.py (Updated v1.1)
import os
import sqlite3
import struct
import time
import numpy as np
import pandas as pd
from kneed import KneeLocator
from typing import List, Dict, Any, Tuple

# --- OPTIONAL IMPORTS ---
try:
    from ibm_watsonx_ai.foundation_models import Embeddings
    from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
except ImportError:
    pass

try:
    import google.generativeai as genai
except ImportError:
    pass

class CitrusEngine:
    def __init__(self, provider="gemini", api_key=None, project_id=None, ibm_url=None, verbose=True):
        self.provider = provider.lower()
        self.api_key = api_key
        self.db_connection = None
        
        # IBM Configuration
        self.ibm_model_id = "ibm/slate-125m-english-rtrvr-v2"
        self.ibm_project_id = project_id
        self.ibm_url = ibm_url
        
        # Gemini Configuration
        self.gemini_model = "models/gemini-embedding-001" # Default Fallback
        
        if self.provider == "gemini" and self.api_key and self.api_key != "placeholder":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            # --- AUTO-DETECT MODEL ---
            try:
                # Find all available embedding models
                available = [m.name for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
                
                if verbose: print(f"🔎 Available Gemini Models: {available}")

                # Preference Logic: 004 (New) > 001 (Old)
                if 'models/text-embedding-004' in available:
                    self.gemini_model = 'models/text-embedding-004'
                elif 'models/gemini-embedding-001' in available:
                    self.gemini_model = 'models/gemini-embedding-001'
                elif 'models/embedding-001' in available:
                    self.gemini_model = 'models/embedding-001'
                
            except Exception as e:
                if verbose: print(f"⚠️ Warning: Could not list Gemini models. Defaulting to {self.gemini_model}")

        if verbose:
            print(f"🍊 CITRUS Engine Initialized via {self.provider.upper()}")
            if self.provider == "gemini":
                print(f"   - Target Model: {self.gemini_model}")

    def _embed_gemini(self, texts: List[str]) -> List[List[float]]:
        """
        Gemini implementation with Legacy Model Support.
        """
        import google.generativeai as genai
        import time
        
        max_retries = 3
        # Clean inputs: remove empty strings
        clean_texts = [t if t.strip() else "Empty content" for t in texts]
        
        for attempt in range(max_retries):
            try:
                # Base arguments
                kwargs = {
                    "model": self.gemini_model,
                    "content": clean_texts,
                    "task_type": "retrieval_document"
                }
                
                # CRITICAL FIX: Only send 'output_dimensionality' if using the NEW model.
                # The old model (gemini-embedding-001) is natively 768, so we don't need this arg.
                # Sending it to the old model causes 404/Invalid Argument errors.
                if "004" in self.gemini_model:
                    kwargs["output_dimensionality"] = 768

                result = genai.embed_content(**kwargs)
                return result['embedding']
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"❌ Gemini Error on {self.gemini_model}: {e}")
                    raise e
                
                # Exponential backoff
                time.sleep(2 * (attempt + 1))

    # --- DATABASE OPERATIONS (sqlite-vec) ---
    
    def connect_db(self, db_path="embeddings_db.sqlite"):
        """Connects to the vector database generated in Phase I."""
        self.db_connection = sqlite3.connect(db_path)
        self.db_connection.enable_load_extension(True)
        # Assuming sqlite-vec is installed in the environment
        try:
            import sqlite_vec
            sqlite_vec.load(self.db_connection)
        except:
            # Fallback for environments where extension loading is manual
            pass 
        self.db_connection.enable_load_extension(False)
        print(f"🔌 Connected to Vector DB: {db_path}")

    def process_scopus_files(self, file_paths: List[str]) -> Tuple[pd.DataFrame, bool]:
        """
        Reads Scopus CSVs, normalizes headers, and prepares text for embedding.
        Returns: (Processed DataFrame, Is_Over_10k_Flag)
        """
        dfs = []
        print(f"📂 Processing {len(file_paths)} file(s)...")
        
        for file in file_paths:
            try:
                # Scopus CSVs sometimes have mixed types, low_memory=False handles it
                temp_df = pd.read_csv(file, on_bad_lines='skip', low_memory=False)
                dfs.append(temp_df)
            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")

        if not dfs:
            raise ValueError("No valid CSV files loaded.")

        # Combine all chunks
        df = pd.concat(dfs, ignore_index=True)
        
        # 1. Standardize Columns (Scopus formatting)
        # Ensure we have the basics even if columns are missing
        required_cols = ['Title', 'Abstract', 'Year', 'DOI', 'Authors']
        for col in required_cols:
            if col not in df.columns:
                df[col] = "" # Fill missing columns with empty strings

        # 2. Data Cleaning (Crucial for Embedding)
        # Fill NaNs with empty strings to prevent concatenation errors
        df['Title'] = df['Title'].fillna("Untitled")
        df['Abstract'] = df['Abstract'].fillna("No abstract available.")
        
        # 3. Create the Embedding Field (The Semantic Payload)
        # Structure: "Title. Abstract"
        df['text_to_embed'] = (
            df['Title'].astype(str).str.strip() + ". " + 
            df['Abstract'].astype(str).str.strip()
        )

        # 4. Check Volume
        is_large_corpus = len(df) > 15000
        
        return df, is_large_corpus

    def init_database(self, db_path="citrus.sqlite"):
        """
        Creates a fresh SQLite database with Vector Search capabilities.
        Schema matches the CITRUS manuscript requirements.
        """
        if os.path.exists(db_path):
            os.remove(db_path) # Clean slate for new analysis
            
        self.db_connection = sqlite3.connect(db_path)
        self.db_connection.enable_load_extension(True)
        
        # Load sqlite-vec extension
        # Note: In Colab, we usually don't need explicit path if installed via pip
        try:
            import sqlite_vec
            self.db_connection.load_extension(sqlite_vec.loadable_path())
        except Exception as e:
            print(f"⚠️ Warning loading sqlite-vec: {e}")

        self.db_connection.enable_load_extension(False)
        
        # Create Schema
        # We use a 768-dim float vector (Standard for Granite & OpenAI MRL)
        cursor = self.db_connection.cursor()
        cursor.execute("""
            CREATE TABLE papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doi TEXT,
                title TEXT,
                year INTEGER,
                abstract TEXT,
                embedding VECTOR(768)
            )
        """)
        self.db_connection.commit()
        print(f"💽 Database initialized at: {db_path}")

    def save_batch(self, texts: List[str], metadatas: List[Dict], vectors: List[List[float]]):
        """
        Inserts a batch of vectors + metadata into SQLite.
        Handles the binary serialization required by sqlite-vec.
        """
        cursor = self.db_connection.cursor()
        sql = "INSERT INTO papers (doi, title, year, abstract, embedding) VALUES (?, ?, ?, ?, ?)"
        
        batch_data = []
        for meta, vec in zip(metadatas, vectors):
            # Serialize vector to raw bytes (Little Endian Float32)
            vec_blob = struct.pack(f'{len(vec)}f', *vec)
            
            batch_data.append((
                meta.get('DOI', ''),
                meta.get('Title', ''),
                meta.get('Year', 0),
                meta.get('Abstract', ''),
                vec_blob
            ))
            
        cursor.executemany(sql, batch_data)
        self.db_connection.commit()

    def compute_jaccard_matrix(self, clusters: Dict[str, str], top_k=50) -> pd.DataFrame:
        """
        Calculates Pairwise Jaccard Similarity between multiple queries.
        Used to validate if search terms are truly distinct (Orthogonal).
        """
        cluster_ids = {}
        
        # 1. Retrieve Result Sets for all clusters
        for label, query_text in clusters.items():
            # Embed
            vec = self.get_embedding(query_text)
            
            # Search DB (Get only IDs)
            # Serialize for sqlite-vec
            query_blob = struct.pack(f'{len(vec)}f', *vec)
            
            cursor = self.db_connection.cursor()
            results = cursor.execute("""
                SELECT rowid FROM papers 
                WHERE vec_distance_cosine(embedding, ?) < 1.0
                ORDER BY vec_distance_cosine(embedding, ?) ASC
                LIMIT ?
            """, [query_blob, query_blob, top_k]).fetchall()
            
            # Store Set of IDs
            cluster_ids[label] = set([r[0] for r in results])

        # 2. Build Matrix
        labels = list(clusters.keys())
        matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
        
        for row in labels:
            for col in labels:
                s1 = cluster_ids[row]
                s2 = cluster_ids[col]
                
                intersection = len(s1.intersection(s2))
                union = len(s1.union(s2))
                
                # Jaccard Formula
                score = intersection / union if union > 0 else 0
                matrix.loc[row, col] = score
                
        return matrix

    def search_similarity(self, query_vec: List[float], limit=500) -> pd.DataFrame:
        """
        Performs Cosine Similarity search.
        Returns DataFrame with 'similarity' (1 - distance) and metadata.
        """
        # Serialize vector to raw bytes for sqlite-vec
        query_blob = np.array(query_vec, dtype=np.float32).tobytes()
        
        cursor = self.db_connection.cursor()
        
        # Matches logic from Screenshot 6
        sql = """
            SELECT 
                rowid,
                title, 
                year, 
                abstract, 
                vec_distance_cosine(embedding, ?) as distance
            FROM papers 
            ORDER BY distance ASC
            LIMIT ?
        """
        
        results = cursor.execute(sql, [query_blob, limit]).fetchall()
        
        # Format into DF
        data = []
        for r in results:
            data.append({
                "id": r[0],
                "title": r[1],
                "year": r[2],
                "abstract": r[3],
                "distance": r[4],
                "similarity": 1 - r[4] # Convert dist to sim
            })
            
        return pd.DataFrame(data)

    # --- HiLAT MATH (Knee Detection) ---

    def calculate_cliff(self, similarities: List[float]) -> Tuple[int, float]:
        """
        Identifies the mathematical 'Elbow' in the decay curve.
        Returns: (rank_index, similarity_score)
        """
        x = range(len(similarities))
        y = similarities
        
        # The logic from Manuscript Section 3.3.2
        kneedle = KneeLocator(x, y, S=1.0, curve="convex", direction="decreasing")
        
        return kneedle.knee, kneedle.knee_y

    def audit_drop_zone(self, df_results: pd.DataFrame, cutoff_rank: int, scope=5):
        """
        Returns the papers immediately rejected by the cutoff.
        Used for the 'Confirm' button logic.
        """
        # Get rows from rank+1 to rank+scope
        drop_zone = df_results.iloc[cutoff_rank : cutoff_rank+scope].copy()
        return drop_zone[['title', 'similarity']]

    # --- JACCARD ORTHOGONALITY ---
    
    def check_orthogonality(self, clusters: Dict[str, str]) -> pd.DataFrame:
        """
        Generates the Jaccard Heatmap data.
        """
        # 1. Retrieve IDs for all clusters
        cluster_ids = {}
        for label, query in clusters.items():
            vec = self.get_embedding(query)
            df = self.search_similarity(vec, limit=50) # Top 50 as per paper
            cluster_ids[label] = set(df['id'].tolist())
            
        # 2. Calculate Matrix
        labels = list(clusters.keys())
        matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
        
        for row in labels:
            for col in labels:
                s1 = cluster_ids[row]
                s2 = cluster_ids[col]
                intersection = len(s1.intersection(s2))
                union = len(s1.union(s2))
                score = intersection / union if union > 0 else 0
                matrix.loc[row, col] = score
                
        return matrix
    
    # --- FINAL HARVEST ---
    
    def harvest_papers(self, cluster_cutoffs: Dict[str, int]) -> Tuple[pd.DataFrame, str]:
        """
        Retrieves the final list of papers based on confirmed cut-offs.
        Returns: (DataFrame of Papers, Audit Report Text)
        """
        all_frames = []
        report_lines = ["CITRUS METHODOLOGY AUDIT LOG", "="*30]
        
        for label, cutoff in cluster_cutoffs.items():
            if cutoff <= 0:
                report_lines.append(f"Cluster '{label}': SKIPPED (n=0)")
                continue
                
            # Retrieve Top N
            # (Re-using search logic but getting full metadata)
            # In production, cache the query vector to avoid re-embedding
            vec = self.get_embedding(active_clusters[label]) # Assuming active_clusters is passed or stored
            
            df = self.search_similarity(vec, limit=cutoff)
            df['Cluster_Source'] = label
            df['Rank_in_Cluster'] = df.index + 1
            all_frames.append(df)
            
            report_lines.append(f"Cluster '{label}': Retrieved Top {cutoff} papers.")

        # Merge & Deduplicate
        if not all_frames:
            return pd.DataFrame(), "\n".join(report_lines)
            
        full_df = pd.concat(all_frames)
        
        # Deduplication Logic
        # Keep the instance with the HIGHEST similarity (if a paper appears in multiple clusters)
        final_df = full_df.sort_values('similarity', ascending=False).drop_duplicates(subset=['title'])
        
        report_lines.append("="*30)
        report_lines.append(f"Total Raw Candidates: {len(full_df)}")
        report_lines.append(f"Unique Papers (Post-Dedup): {len(final_df)}")
        
        return final_df, "\n".join(report_lines)