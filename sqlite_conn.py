import sqlite3
from typing import Dict

from numpy import place


class sqlite_db():

    def __init__(self, path):
       self.path = path


    def conn_to_db(self):
        #Connecting to sqlite
        self.conn = sqlite3.connect(self.path)
        #Creating a cursor object using the cursor() method
        self.cursor = self.conn.cursor()

    def create_db_if_exists(self, table_name: str, schema: Dict):

        execution_string = f"""CREATE TABLE IF NOT EXISTS {table_name} ("""

        for key, data_type in schema.items():
            execution_string += f"{key} {data_type},"
         
        execution_string = execution_string[:-1] + ")" 
        print(execution_string)
        self.cursor.execute(execution_string)

        # Commit your changes in the database
        self.conn.commit()

    def append_row_to_table(self, input_dict: Dict, table_name: str):

        columns = ', '.join(input_dict.keys())
        placeholders = ':'+', :'.join(input_dict.keys())
        query = f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})'
        print(query)
        self.cursor.execute(query, input_dict)
        self.conn.commit()

    def write_to_db(self):
        pass

    def append_to_db(self):
        pass

    def close_conn(self):
        self.conn.close()

    