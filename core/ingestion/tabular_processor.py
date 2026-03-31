import pandas as pd
import sqlite3
import re
from pathlib import Path
from core.config import SQL_DB_PATH

class TabularProcessor:
    def __init__(self):
        self.db_path = SQL_DB_PATH

    def _clean_string(self, text):
        """Cleans strings to be safe for SQL table and column names."""
        clean = re.sub(r'\W+', '_', str(text).strip()).lower()
        # Ensure it doesn't start with a number
        if clean and clean[0].isdigit():
            clean = "col_" + clean
        return clean

    def process_file(self, file_path, status_callback=None):
        """
        Reads a CSV or Excel file and converts it into a SQLite table.
        The table name is derived from the file name.
        """
        ext = Path(file_path).suffix.lower()
        filename = Path(file_path).name
        
        if status_callback:
            status_callback(f"Loading {filename} into memory...")

        try:
            # 1. Load the file into a Pandas DataFrame
            if ext == '.csv':
                df = pd.read_csv(file_path)
            elif ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported tabular format: {ext}")

            # 2. Clean Column Names for SQL safety
            df.columns = [self._clean_string(col) for col in df.columns]
            
            # 3. Generate a safe SQL Table Name from the filename
            table_name = self._clean_string(Path(file_path).stem)

            if status_callback:
                status_callback(f"Writing {len(df)} rows to SQL table '{table_name}'...")

            # 4. Connect to SQLite and write the data
            with sqlite3.connect(self.db_path) as conn:
                # if_exists='replace' means if they upload an updated CSV, it overwrites the old table
                df.to_sql(table_name, conn, if_exists='replace', index=False)

            return {
                "status": "success", 
                "table_name": table_name, 
                "columns": list(df.columns), 
                "row_count": len(df)
            }

        except Exception as e:
            print(f"⚠️ Error processing tabular data: {e}")
            return {"status": "error", "message": str(e)}