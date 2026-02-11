# ==============================================================================
# CITRUS: The Rehydration Protocol (Copyright Compliance Tool)
# ==============================================================================
# 
# PURPOSE:
# This script "rehydrates" the sanitized CITRUS epistemic artifact by fetching
# the missing abstracts from Scopus. It is designed for researchers who possess
# their own valid Scopus API credentials (institutional access).
#
# LOGIC:
# 1. Connects to the public 'citrus_epistemic_artifact.sqlite' (Vectors + Metadata).
# 2. Identifies records where the 'abstract' column is missing or NULL.
# 3. Uses the DOI to query the Scopus API via `pybliometrics`.
# 4. Updates the local SQLite database with the retrieved text.
#
# PREREQUISITES:
# 1. pip install pybliometrics sqlite-vec
# 2. A valid Scopus API Key (https://dev.elsevier.com/)
# ==============================================================================

import sqlite3
import time
from tqdm import tqdm # Progress bar
from pybliometrics.scopus import AbstractRetrieval, init
from pybliometrics.scopus.exception import Scopus404Error, Scopus429Error

# --- CONFIGURATION ---
DB_PATH = 'citrus_epistemic_artifact.sqlite' # The sanitized file
BATCH_SIZE = 50 # Commit to DB every 50 records to save progress

def setup_environment():
    """Ensures pybliometrics is configured with a key."""
    print("Checking Scopus API Configuration...")
    try:
        init() # This will prompt for API Key if not already set
    except Exception as e:
        print(f"Error: {e}")
        print("Please configure pybliometrics with your API Key.")
        return False
    return True

def rehydrate():
    if not setup_environment():
        return

    print(f"🔌 Connecting to artifact: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Check if 'abstract' column exists (it might have been dropped)
    # If not, create it.
    cursor.execute("PRAGMA table_info(papers)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'abstract' not in columns:
        print("Creating 'abstract' column structure...")
        cursor.execute("ALTER TABLE papers ADD COLUMN abstract TEXT")
        conn.commit()

    # 2. Find Dehydrated Records (DOIs with no abstract)
    print("🔍 Scanning for dehydrated records...")
    cursor.execute("SELECT rowid, DOI FROM papers WHERE abstract IS NULL OR abstract = ''")
    target_rows = cursor.fetchall()
    
    total_missing = len(target_rows)
    print(f"💦 Found {total_missing} records requiring rehydration.")
    
    if total_missing == 0:
        print("✅ Artifact is fully hydrated. No action needed.")
        return

    # 3. The Fetch Loop
    print("🚀 Starting Rehydration Process (press Ctrl+C to pause)...")
    success_count = 0
    error_count = 0
    
    # We use TQDM for a professional progress bar
    pbar = tqdm(target_rows, desc="Fetching Scopus Abstracts", unit="paper")
    
    for index, (row_id, doi) in enumerate(pbar):
        if not doi or doi == "None":
            continue
            
        try:
            # The API Call
            # view='FULL' ensures we get the abstract
            ab = AbstractRetrieval(doi, view='FULL', refresh=False)
            
            # Extract Abstract
            abstract_text = ab.description
            
            if abstract_text:
                # Update DB
                cursor.execute("UPDATE papers SET abstract = ? WHERE rowid = ?", (abstract_text, row_id))
                success_count += 1
            else:
                error_count += 1
                
        except Scopus404Error:
            # DOI not found in Scopus (common with pre-prints or bad metadata)
            # We mark it as 'NOT FOUND' so we don't retry forever
            cursor.execute("UPDATE papers SET abstract = 'SCOPUS_404_NOT_FOUND' WHERE rowid = ?", (row_id,))
            error_count += 1
        except Scopus429Error:
            print("\n⚠️ API Rate Limit Reached. Sleeping for 5 seconds...")
            time.sleep(5)
        except Exception as e:
            # Other network errors
            error_count += 1
            
        # Batch Commit
        if index % BATCH_SIZE == 0:
            conn.commit()
            
    # Final Commit
    conn.commit()
    conn.close()
    
    print("\n" + "="*50)
    print("✅ REHYDRATION COMPLETE")
    print(f"   Successfully Retrieved: {success_count}")
    print(f"   Errors / Not Found:     {error_count}")
    print("   The database now contains full text for local analysis.")
    print("="*50)

if __name__ == "__main__":
    rehydrate()