import pymysql

class mysql_moudle(object):

    def __init__(self, host="192.168.254.20", port=3306, user="root",
                password="123456", database="WPF",charset="utf8"):


        self.host = host
        self.user = user
        self.port = port
        self.__password = password
        self.database = database
        #self.connector = self.connect()
        self.charset = charset

    def mysql_create_table(self):

        connect = pymysql.connect(
            user=self.user,
            password=self.__password,
            host=self.host,
            port=self.port,
            db=self.database,
            charset=self.charset
        )
        conn = connect.cursor()  # 创建操作游标

        #                          创建表
        conn.execute("drop table if exists url_kmeans_data")  # 如果表存在则删除
        #url, server_name, common_name, max_tcp_number, min_tcp_number, mean_tcp_number, std_tcp_number, max_sum_packet, min_sum_packet, mean_sum_packet, std_sum_packet, max_sum_ssl, min_sum_ssl, mean_sum_ssl, std_sum_ssl, url_counts
        # 使用预处理语句创建表
        sql = """CREATE TABLE url_kmeans_data (
                 tcp_number FLOAT,
                 sum_packet FLOAT,
                 sum_ssl FLOAT,
                 packet_number_0 FLOAT,
                 ssl_number_0 FLOAT,
                 min_ssl_size_0 FLOAT,
                 max_ssl_size_0 FLOAT,
                 mean_ssl_size_0 FLOAT,
                 median_ssl_size_0 FLOAT,
                 mode_ssl_size_0 FLOAT,
                 var_ssl_size_0 FLOAT,
                 perc_25_0 FLOAT,
                 perc_75_0 FLOAT,
                 cluster INT,
                 url CHAR(255) NOT NULL)
                 """
        conn.execute(sql)  # 创建表
        conn.close()  # 关闭游标连接

if __name__ == '__main__':
    mysql_moudle().mysql_create_table()