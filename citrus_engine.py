"""
CITRUS ENGINE (v1.2)
Core logic for: From Static Keywords to Adaptive Vectors
Author: Wisudanto C. Suntoyo
License: MIT
"""

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
        
        # Gemini Configuration (Dynamic Discovery)
        self.gemini_model = "models/gemini-embedding-001" # Default Fallback
        
        if self.provider == "gemini" and self.api_key and self.api_key != "placeholder":
            genai.configure(api_key=self.api_key)
            
            # --- AUTO-DETECT AVAILABLE MODELS ---
            try:
                # List models that support embeddings
                models = [m for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
                model_names = [m.name for m in models]
                
                if verbose:
                    print(f"🔎 Available Gemini Models: {model_names}")

                # Priority: 004 (Newest) -> 001 (Legacy/Stable)
                if 'models/text-embedding-004' in model_names:
                    self.gemini_model = 'models/text-embedding-004'
                elif 'models/gemini-embedding-001' in model_names:
                    self.gemini_model = 'models/gemini-embedding-001'
                elif 'models/embedding-001' in model_names:
                    self.gemini_model = 'models/embedding-001'
                    
            except Exception as e:
                if verbose: print(f"⚠️ Warning: Could not list Gemini models ({e}). Using default.")

        if verbose:
            print(f"🍊 CITRUS Engine Initialized via {self.provider.upper()}")
            if self.provider == "gemini":
                print(f"   - Target Model: {self.gemini_model}")

    # --- THE MISSING ROUTER METHOD ---
    def get_embedding(self, text_input: Any) -> List[List[float]]:
        """
        Public method called by Dashboard. Routes to specific provider logic.
        """
        # Normalize input to list
        is_single = isinstance(text_input, str)
        texts = [text_input] if is_single else text_input

        if self.provider == "ibm":
            vectors = self._embed_ibm(texts)
        elif self.provider == "gemini":
            vectors = self._embed_gemini(texts)
        elif self.provider == "openai":
            vectors = self._embed_openai(texts)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
            
        return vectors[0] if is_single else vectors

    # --- PROVIDER IMPLEMENTATIONS ---

    def _embed_ibm(self, texts: List[str]) -> List[List[float]]:
        embed_params = {
            EmbedParams.TRUNCATE_INPUT_TOKENS: 512,
            EmbedParams.RETURN_OPTIONS: {'input_text': False}
        }
        model = Embeddings(
            model_id=self.ibm_model_id,
            params=embed_params,
            credentials={"url": self.ibm_url, "apikey": self.api_key},
            project_id=self.ibm_project_id
        )
        return model.embed_documents(texts)

    def _embed_gemini(self, texts: List[str]) -> List[List[float]]:
        """
        Gemini implementation with Legacy Model Support.
        """
        max_retries = 3
        # Clean inputs: remove empty strings which crash Gemini
        clean_texts = [t if t.strip() else "Empty content" for t in texts]
        
        for attempt in range(max_retries):
            try:
                # Base arguments
                kwargs = {
                    "model": self.gemini_model,
                    "content": clean_texts,
                    "task_type": "retrieval_document"
                }
                
                # CRITICAL: Only send 'output_dimensionality' if using the NEW model (004)
                if "004" in self.gemini_model:
                    kwargs["output_dimensionality"] = 768

                result = genai.embed_content(**kwargs)
                return result['embedding']
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"❌ Gemini Error: {e}")
                    raise e
                time.sleep(2 * (attempt + 1))

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
            dimensions=768
        )
        return [d.embedding for d in response.data]

    # --- DATABASE MANAGEMENT (Updated Schema) ---

    def init_database(self, db_path="citrus.sqlite"):
        if os.path.exists(db_path):
            os.remove(db_path)
            
        self.db_connection = sqlite3.connect(db_path)
        self.db_connection.enable_load_extension(True)
        try:
            import sqlite_vec
            self.db_connection.load_extension(sqlite_vec.loadable_path())
        except:
            pass 
        self.db_connection.enable_load_extension(False)
        
        cursor = self.db_connection.cursor()
        cursor.execute("""
            CREATE TABLE papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doi TEXT,
                title TEXT,
                year INTEGER,
                abstract TEXT,
                authors TEXT,
                source TEXT,
                embedding VECTOR(768)
            )
        """)
        self.db_connection.commit()
        print(f"💽 Database initialized at: {db_path}")

    def save_batch(self, texts: List[str], metadatas: List[Dict], vectors: List[List[float]]):
        cursor = self.db_connection.cursor()
        # ADDED: authors, source
        sql = "INSERT INTO papers (doi, title, year, abstract, authors, source, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)"
        
        batch_data = []
        for meta, vec in zip(metadatas, vectors):
            vec_blob = struct.pack(f'{len(vec)}f', *vec)
            batch_data.append((
                meta.get('DOI', ''),
                meta.get('Title', ''),
                meta.get('Year', 0),
                meta.get('Abstract', ''),
                meta.get('Authors', ''), 
                meta.get('Source title', ''), 
                vec_blob
            ))
        cursor.executemany(sql, batch_data)
        self.db_connection.commit()

    # --- ANALYSIS & DATA ---

    def process_scopus_files(self, file_paths: List[str]) -> Tuple[pd.DataFrame, bool]:
        dfs = []
        for file in file_paths:
            try:
                temp_df = pd.read_csv(file, on_bad_lines='skip', low_memory=False)
                dfs.append(temp_df)
            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")

        if not dfs: return pd.DataFrame(), False

        df = pd.concat(dfs, ignore_index=True)
        
        # Normalize
        for col in ['Title', 'Abstract', 'Year', 'DOI', 'Authors']:
            if col not in df.columns: df[col] = ""

        df['Title'] = df['Title'].fillna("Untitled")
        df['Abstract'] = df['Abstract'].fillna("No abstract available.")
        
        df['text_to_embed'] = (
            df['Title'].astype(str).str.strip() + ". " + 
            df['Abstract'].astype(str).str.strip()
        )
        
        return df, len(df) > 15000

    def search_similarity(self, query_vec: List[float], limit=500) -> pd.DataFrame:
        query_blob = np.array(query_vec, dtype=np.float32).tobytes()
        cursor = self.db_connection.cursor()
        # ADDED: authors, source retrieval
        sql = """
            SELECT rowid, title, year, abstract, authors, source, doi, vec_distance_cosine(embedding, ?) as distance
            FROM papers ORDER BY distance ASC LIMIT ?
        """
        results = cursor.execute(sql, [query_blob, limit]).fetchall()
        
        data = []
        for r in results:
            data.append({
                "id": r[0], "title": r[1], "year": r[2], "abstract": r[3],
                "authors": r[4], "source": r[5], "doi": r[6],
                "distance": r[7], "similarity": 1 - r[7]
            })
        return pd.DataFrame(data)

    def calculate_cliff(self, similarities: List[float]) -> Tuple[int, float]:
        x = range(len(similarities))
        kneedle = KneeLocator(x, similarities, S=1.0, curve="convex", direction="decreasing")
        return kneedle.knee, kneedle.knee_y

    def audit_drop_zone(self, df_results: pd.DataFrame, cutoff_rank: int, scope=5):
        return df_results.iloc[cutoff_rank : cutoff_rank+scope][['title', 'similarity']]

    def compute_jaccard_matrix(self, clusters: Dict[str, str], top_k=50) -> pd.DataFrame:
        cluster_ids = {}
        for label, query_text in clusters.items():
            vec = self.get_embedding(query_text)
            query_blob = struct.pack(f'{len(vec)}f', *vec)
            
            cursor = self.db_connection.cursor()
            results = cursor.execute("""
                SELECT rowid FROM papers 
                WHERE vec_distance_cosine(embedding, ?) < 1.0
                ORDER BY vec_distance_cosine(embedding, ?) ASC
                LIMIT ?
            """, [query_blob, query_blob, top_k]).fetchall()
            cluster_ids[label] = set([r[0] for r in results])

        labels = list(clusters.keys())
        matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
        
        for row in labels:
            for col in labels:
                s1, s2 = cluster_ids[row], cluster_ids[col]
                intersection = len(s1.intersection(s2))
                union = len(s1.union(s2))
                matrix.loc[row, col] = intersection / union if union > 0 else 0
                
        return matrix
    
    def harvest_papers(self, cluster_cutoffs: Dict[str, int], active_clusters: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """
        Returns: (Included_DF, Excluded_DF, Audit_Log_String)
        """
        included_frames = []
        excluded_frames = []
        report_lines = ["CITRUS METHODOLOGY AUDIT LOG", "="*30]
        
        for label, cutoff in cluster_cutoffs.items():
            if cutoff <= 0: continue
            
            # Re-run search to get full metadata
            vec = self.get_embedding(active_clusters[label])
            
            # 1. Get INCLUDED (Rank 1 to n)
            df_inc = self.search_similarity(vec, limit=cutoff)
            df_inc['Cluster_Source'] = label
            df_inc['Status'] = "INCLUDED"
            df_inc['Rank'] = df_inc.index + 1
            included_frames.append(df_inc)
            
            # 2. Get EXCLUDED (Rank n+1 to n+5) - The "Border Patrol"
            # We search slightly deeper (cutoff + 5) and slice the tail
            df_deep = self.search_similarity(vec, limit=cutoff + 5)
            df_exc = df_deep.iloc[cutoff:].copy()
            df_exc['Cluster_Source'] = label
            df_exc['Status'] = "EXCLUDED (Border Audit)"
            df_exc['Rank'] = df_exc.index + 1
            excluded_frames.append(df_exc)
            
            report_lines.append(f"Cluster '{label}': Included Top {cutoff} | Audited Next {len(df_exc)}")

        # Merge & Deduplicate Included
        if not included_frames: return pd.DataFrame(), pd.DataFrame(), "No papers selected."
            
        final_inc = pd.concat(included_frames).sort_values('similarity', ascending=False).drop_duplicates(subset=['title'])
        final_exc = pd.concat(excluded_frames) 
        
        # Add DOI Links
        final_inc['url'] = final_inc['doi'].apply(lambda x: f"https://doi.org/{x}" if x else "")
        final_exc['url'] = final_exc['doi'].apply(lambda x: f"https://doi.org/{x}" if x else "")

        report_lines.append("="*30)
        report_lines.append(f"Total Unique Included: {len(final_inc)}")
        
        return final_inc, final_exc, "\n".join(report_lines)