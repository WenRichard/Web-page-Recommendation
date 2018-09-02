import pymysql
import pandas as pd


# 连接数据库
def connector(host, user, password, database):
    db = pymysql.connect(host, user, password, database)
    return db


# 断开数据库
def closer(db):
    db.close()
    return "the database has been closed."


# 读取数据：从csv文本文件
# url, date, features(1~n), labels(1~m)
class CSVReader(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.csv_to_df()

    def csv_to_df(self):
        return pd.read_csv(self.file_path)


# 数据库连接器类
class MySQLReader(object):
    '''一个连接MySQL数据库的对象'''

    def __int__(self, host="localhost", user="root",
                password="broadtech", database="world"):
        self.database_connector = pymysql.connect(host, user, password, database)

    def connector(self, host, user, password, database):
        db = pymysql.connect(host, user, password, database)
        return db

    def close(self):
        self.database_connector.close()

    def exec(self, sql_exp=""):
        '''执行一段sql语句并获取全部返回结果'''
        cursor = self.database_connector.cursor()
        cursor.execute(sql_exp)
        data = cursor.fetchall()
        if data is ():
            print("Query result is empty.")
            return None
        return data

    def get_table(self, table_name="None"):
        '''获取一张表的全部内容，转换为pd.DataFrame对象'''
        data = self.exec("select * from %s;" % table_name)
        if data is not None:
            data = list(map(list, data))
            data = pd.DataFrame(data)
            del data[0]
            return data
        else:
            print("Get an empty table.")
            return None




if __name__ == "__main__":
    d = MySQLReader()
    d.__int__(database="WPF")
    data = d.exec("select * from visit_url_features limit 1;")
    print(data)
