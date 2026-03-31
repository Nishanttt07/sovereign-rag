import sqlite3
import re
from core.config import SQL_DB_PATH
from core.models.llm import LLMEngine

class SQLAgent:
    def __init__(self):
        self.db_path = SQL_DB_PATH
        self.llm = LLMEngine()

    def _get_database_schema(self):
        """Extracts the table names and column names from the SQLite database."""
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
        """Generates and executes a SQL query based on the user's question."""
        schema = self._get_database_schema()
        
        if "Database is empty" in schema or "Error" in schema:
            return None # No tabular data available
            
        # 1. Ask the LLM to write the SQL query
        prompt = f"""You are a SQLite expert. 
        Given the following database schema:
        {schema}
        
        Write a SQL query to answer this user question: "{user_question}"
        
        RULES:
        1. Return ONLY the raw SQL query. No explanations, no markdown blocks, no formatting.
        2. ONLY write SELECT statements. Never UPDATE, DELETE, or DROP.
        3. Make sure table and column names exactly match the schema.
        """
        
        messages = [{"role": "system", "content": prompt}]
        
        # We use standard generate (not stream) because we need the full SQL string instantly
        llm_response = ""
        for chunk in self.llm.chat_stream(messages):
            llm_response += chunk
            
        # 2. Clean the response (in case the LLM ignored rules and added markdown like ```sql)
        clean_sql = re.sub(r'```sql|```', '', llm_response).strip()
        
        # 3. Execute the Query
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Security Check: Prevent prompt injection deletions
                if not clean_sql.upper().startswith("SELECT"):
                    return "[SQL AGENT BLOCKED: Only SELECT queries are allowed for safety]"
                    
                cursor.execute(clean_sql)
                results = cursor.fetchall()
                
                if not results:
                    return f"[SQL Agent searched the database but found 0 results for this query. SQL used: {clean_sql}]"
                    
                # Format the raw data into a readable string for the final LLM
                columns = [description[0] for description in cursor.description]
                data_string = f"Database Table Results:\n{columns}\n"
                for row in results:
                    data_string += f"{row}\n"
                    
                return data_string

        except Exception as e:
            return f"[SQL Agent failed to execute query: {clean_sql}. Error: {e}]"