
from sqlalchemy import create_engine
import pandas as pd

class MySqlDataFrame(object):

    def __init__(self, host="192.168.254.20", port="3306", user="root",
                password="123456", database="WPF"):
        self.host = host
        self.user = user
        self.port = port
        self.__password = password
        self.database = database
        self.connector = self.connect()

    def connect(self):
        engine = create_engine('mysql+pymysql://{root}:{password}@{host}:{port}/{database}' \
                               .format(root=self.user, password=self.__password,
                                       host=self.host, port=self.port, database=self.database))
        conn = engine.connect()
        return conn

    def read_table(self, table_name):
        return pd.read_sql_table(table_name=table_name, con=self.connector)

    def read_query(self, sql_expr):
        return pd.read_sql(sql_expr, con=self.connector)