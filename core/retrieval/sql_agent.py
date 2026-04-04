import sqlite3
import re
from core.config import SQL_DB_PATH
from core.models.llm import LLMEngine

class SQLAgent:
    def __init__(self):
        self.db_path = SQL_DB_PATH
        # 🔥 We explicitly call the 3B coding genius for SQL tasks.
        # This keeps VRAM usage low and accuracy extremely high.
        self.llm = LLMEngine(model_name="qwen2.5-coder:3b")

    def _get_database_schema(self):
        """Extracts the table names and column definitions from SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                if not tables: 
                    return "Database is empty."
                    
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
        """Generates SQL from the user query, executes it, and returns a Markdown table."""
        schema = self._get_database_schema()
        
        if "Database is empty" in schema or "Error" in schema: 
            return None
            
        # 1. GENERATE SQL USING QWEN CODER WITH DATE NORMALIZATION
        prompt = f"""You are an elite Data Engineer and SQLite Expert. 
        Given this database schema:
        {schema}
        
        CRITICAL RULES:
        1. Dates in the database are stored in YYYY-MM-DD format. If the user provides a date in a different format, you MUST convert it to YYYY-MM-DD.
        2. Use the exact column names provided in the schema. Use LIKE '%keyword%' for text searches to be safe.
        3. If the user asks for the "last", "latest", "recent", or "bottom" records, you MUST use an ORDER BY clause (e.g., ORDER BY rowid DESC, or ORDER BY the ID column DESC) with a LIMIT clause.
        
        User Question: {user_question}
        
        Write ONLY the valid SQLite query to answer the question. 
        Do not explain your code. Do not include introductory text. 
        Start your response immediately with the word SELECT."""
        
        messages = [{"role": "system", "content": prompt}]
        llm_response = ""
        
        for chunk in self.llm.chat_stream(messages):
            llm_response += chunk
            
        # Clean the output (in case the model wraps it in markdown code blocks)
        clean_sql = re.sub(r'```sql|```', '', llm_response).strip()
        print(f"\n[🤖 SQL AGENT (Qwen)] Executing Query -> {clean_sql}\n") 

        # 2. EXECUTE QUERY & FORMAT AS MARKDOWN TABLE
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Security check: Only allow SELECT statements
                if not clean_sql.upper().startswith("SELECT"): 
                    return "Error: Only SELECT queries are permitted for safety."
                
                cursor.execute(clean_sql)
                results = cursor.fetchall()
                
                if not results: 
                    return "No matching records found in the database."
                
                # Extract column names from the cursor description
                columns = [description[0] for description in cursor.description]
                
                # Build the Markdown Table
                header = "| " + " | ".join(columns) + " |"
                separator = "| " + " | ".join(["---"] * len(columns)) + " |"
                body = ""
                for row in results:
                    body += "| " + " | ".join(str(item) for item in row) + " |\n"
                
                return f"\n{header}\n{separator}\n{body}\n"

        except Exception as e:
            return f"SQL Execution Error: {e}"