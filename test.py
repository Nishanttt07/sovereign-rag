import lancedb
import pandas as pd
from core.config import DB_DIR

def inspect():
    print(f"📂 Opening Database at: {DB_DIR}")
    try:
        db = lancedb.connect(str(DB_DIR))
        if "vectors" not in db.table_names():
            print("❌ Error: 'vectors' table does not exist. Index is empty.")
            return
            
        table = db.open_table("vectors")
        # Fetch all data (up to 10,000 chunks)
        df = table.search().limit(10000).to_pandas()
        
        print(f"✅ Total Chunks Found: {len(df)}")
        
        # Filter for 'tides'
        tide_chunks = df[df['text'].str.contains("tide", case=False, na=False)]
        
        if not tide_chunks.empty:
            print("\n✅ FOUND 'TIDE' DATA:")
            for _, row in tide_chunks.iterrows():
                print(f" - Page {row['metadata']['page']}: {row['text'][:100]}...")
        else:
            print("\n❌ 'TIDE' NOT FOUND in database.")
            print("   (This confirms the text was not in the uploaded PDF)")

        print("\n📊 Sample of stored topics:")
        print(df[['text', 'metadata']].head(5))

    except Exception as e:
        print(f"⚠️ Inspection Error: {e}")

if __name__ == "__main__":
    inspect()