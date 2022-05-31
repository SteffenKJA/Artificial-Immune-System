import sqlite3


class sqlite_db():

    def __init__(self, path):
       self.path = path


    def conn_to_db(self):
        #Connecting to sqlite
        self.conn = sqlite3.connect(self.path)
        #Creating a cursor object using the cursor() method
        self.cursor = self.conn.cursor()

    def create_db(self, table_name: str, feature_dict: list):

        execution_string = """CREATE TABLE IF NOT EXISTS {table_name}"""

        for key, data_type in feature_dict:
            execution_string += f"{key} {data_type},"
         
        execution_string = execution_string[:-1] + ")" 
        self.cursor.execute(execution_string)

        # Commit your changes in the database
        self.conn.commit()

    def close_conn(self):
        self.conn.close()

    