import sqlite3
import re
from core.config import SQL_DB_PATH
from core.models.llm import LLMEngine

class SQLAgent:
    def __init__(self):
        self.db_path = SQL_DB_PATH
        self.llm = LLMEngine()

    def _get_database_schema(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                if not tables: return "Database is empty."
                schema = []
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    col_details = [f"{col[1]} ({col[2]})" for col in columns]
                    schema.append(f"Table '{table_name}' has columns: {', '.join(col_details)}")
                return "\n".join(schema)
        except Exception as e:
            return f"Error reading schema: {e}"

    def query(self, user_question):
        schema = self._get_database_schema()
        if "Database is empty" in schema or "Error" in schema: return None
            
        # 1. GENERATE SQL
        prompt = f"""You are a SQLite Expert. Given this schema:
        {schema}
        
        Question: {user_question}
        
        Write ONLY the SQL query. Do not explain. Do not use markdown. Start with SELECT."""
        
        messages = [{"role": "system", "content": prompt}]
        llm_response = ""
        for chunk in self.llm.chat_stream(messages):
            llm_response += chunk
            
        clean_sql = re.sub(r'```sql|```', '', llm_response).strip()
        print(f"DEBUG: SQL Agent executing -> {clean_sql}") # Check your terminal for this!

        # 2. EXECUTE & FORMAT AS MARKDOWN
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if not clean_sql.upper().startswith("SELECT"): return None
                
                cursor.execute(clean_sql)
                results = cursor.fetchall()
                if not results: return "No matching records found in SQL."
                
                columns = [description[0] for description in cursor.description]
                
                # Create a Markdown Table string
                header = "| " + " | ".join(columns) + " |"
                separator = "| " + " | ".join(["---"] * len(columns)) + " |"
                body = ""
                for row in results:
                    body += "| " + " | ".join(str(item) for item in row) + " |\n"
                
                return f"\n{header}\n{separator}\n{body}\n"

        except Exception as e:
            return f"SQL Error: {e}"