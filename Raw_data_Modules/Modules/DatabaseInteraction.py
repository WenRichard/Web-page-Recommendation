from Modules import DatabaseConnector as dc

if __name__ == "__main__":
    d = dc.MySQLReader()
    d.__int__(database="WPF")
    # g = open("scripts/insert_into_visit_urls.sql", "w")
    g = open("scripts/insert_into_visit_url_features.sql", "w")
    g.write("truncate table visit_url_features;\n")
    # g.write("truncate table visit_urls;\n")
    # with open("data_temp/url_list.csv") as f:
    with open("data_temp/dataset3.csv") as f:
        for line in f:
            features = line.rstrip("\n").split(",")
            for i in range(2, 17):
                features[i] = "'%s'" % features[i]
            features[-2:] = ["'%s'" % features[-2], "'%s'" % features[-1]]

            sql_line = ",".join(features)
            # id, timestamp, url, log_file = line.rstrip("\n").split(",")
            sql_line = "insert into visit_url_features values (%s);\n" % sql_line
            # sql_line = "insert into visit_urls values (%s, '%s', '%s', '%s');\n" % (id, timestamp, url, log_file)
            print(sql_line)
            g.write(sql_line)
    g.close()



