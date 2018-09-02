from Modules.preprocessing import URLSample
from Modules.DatabaseConnector import MySQLReader
import numpy as np
from scipy import stats
import pandas as pd
import Modules.constants as constants


class URLSampleFeatures(URLSample):
    def __init__(self, url, log_file):
        super().__init__(log_file)
        self.url = url
        self.TCP_feature_list = self.get_TCP_feature_list()
        self.tcp_len = len(self.TCP_feature_list)

    def extract_url_features(self):

        def fill_blank(x, length, fill_value=0):
            assert isinstance(fill_value, (str, int))
            l = len(x)
            if l >= length:
                return x[:length]
            else:
                xx = [fill_value for i in range(length)]
                for j in range(l):
                    xx[j] = x[j]
                return xx
        features = {}
        names = []
        blank_length = 5 # 占位长度
        features['tcp_number'] = self.tcp_len
        names.append('tcp_number')
        servers = [d['server_name'] for d in self.TCP_feature_list]
        features['main_server'] = fill_blank(list(set(servers)), blank_length, fill_value="None")
        names.extend(['main_server_%d' % i for i in range(blank_length)])
        commons = [d['common_name'] for d in self.TCP_feature_list]
        features['main_common'] = fill_blank(list(set(commons)), blank_length, fill_value="None")
        names.extend(['main_common_%d' % i for i in range(blank_length)])
        ciphers = [d['cipher_suite'] for d in self.TCP_feature_list]
        features['main_cipher'] = fill_blank(list(set(ciphers)), blank_length, fill_value="None")
        names.extend(['main_cipher_%d' % i for i in range(blank_length)])

        place_holder = 80
        empty_ = [0 for i in range(place_holder)]
        packet_number_list = empty_.copy()
        ssl_load_number_list = empty_.copy()
        min_size = empty_.copy()
        max_size = empty_.copy()
        min_ssl_size = empty_.copy()
        max_ssl_size = empty_.copy()
        mean_ssl_size = empty_.copy()
        median_ssl_size = empty_.copy()
        mode_ssl_size = empty_.copy()
        var_ssl_size = empty_.copy()
        perc_25, perc_75 = empty_.copy(), empty_.copy()
        for i in range(self.tcp_len):
            t = self.TCP_feature_list[i]
            packet_number_list[i] = t['packet_number']
            ssl_load_number_list[i] = t['ssl_load_number']
            min_size[i] = t['object_message'][0]
            max_size[i] = t['object_message'][1]
            min_ssl_size[i] = min([i for i in t['ssl_load_message'] if i > 0])
            max_ssl_size[i] = max(t['ssl_load_message'])
            mean = lambda a: sum(a) / len(a)
            mean_ssl_size[i] = mean([i for i in t['ssl_load_message'] if i > 0])
            median_ssl_size[i] = float(np.median([i for i in t['ssl_load_message'] if i > 0]))
            mode_ssl_size[i] = stats.mode([i for i in t['ssl_load_message'] if i > 0])[0][0]
            var_ssl_size[i] = np.var([i for i in t['ssl_load_message'] if i > 0])
            perc_25[i] = np.percentile([i for i in t['ssl_load_message'] if i > 0], 25)
            perc_75[i] = np.percentile([i for i in t['ssl_load_message'] if i > 0], 75)

        features['sum_packet'] = sum(packet_number_list)
        features['sum_ssl'] = sum(ssl_load_number_list)

        features['packets_number'] = packet_number_list
        features['ssl_number'] = ssl_load_number_list

        features['min_ssl_size'] = min_ssl_size
        features['max_ssl_size'] = max_ssl_size
        features['mean_ssl_size'] = mean_ssl_size
        features['median_ssl_size'] = median_ssl_size
        features['mode_ssl_size'] = mode_ssl_size
        features['var_ssl_size'] = var_ssl_size
        features['perc_25'] = perc_25
        features['perc_75'] = perc_75

        names.append('sum_packet')
        names.append('sum_ssl')

        names.extend(['packet_number_%d' %d for d in range(place_holder)])
        names.extend(['ssl_number_%d' %d for d in range(place_holder)])
        names.extend(['min_ssl_size_%d' %d for d in range(place_holder)])
        names.extend(['max_ssl_size_%d' %d for d in range(place_holder)])
        names.extend(['mean_ssl_size_%d' %d for d in range(place_holder)])
        names.extend(['median_ssl_size_%d' %d for d in range(place_holder)])
        names.extend(['mode_ssl_size_%d' %d for d in range(place_holder)])
        names.extend(['var_ssl_size_%d' %d for d in range(place_holder)])
        names.extend(['perc_25_%d' %d for d in range(place_holder)])
        names.extend(['perc_75_%d' %d for d in range(place_holder)])
        # for n in names:
        #     print("'%s'," % n)
        return features

class WebFingerprintSample(object):
    def __init__(self, sample):
        self.sample = sample
        self.keys = constants.url_columns[:-1]
        self.data = pd.DataFrame(list(map(list, data)), columns=constants.visit_url_features_columns)



def flatten_features(f):
    """将多维列表展平以便写入文件"""
    g = []
    g.append(f['tcp_number'])
    g.extend(f['main_server'])
    g.extend(f['main_common'])
    g.extend(f['main_cipher'])
    g.append(f['sum_packet'])
    g.append(f['sum_ssl'])
    g.extend(f['packets_number'])
    g.extend(f['ssl_number'])
    g.extend(f['min_ssl_size'])
    g.extend(f['max_ssl_size'])
    g.extend(f['mean_ssl_size'])
    g.extend(f['median_ssl_size'])
    g.extend(f['mode_ssl_size'])
    g.extend(f['var_ssl_size'])
    g.extend(f['perc_25'])
    g.extend(f['perc_75'])
    return g

if __name__ == "__main__":
    # url = "https://car.autohome.com.cn/price/series-2951.html#pvareaid=103446"
    # log_file = r"data\fpf_data\car.autohome.com.cn\20180726\2018072613\1532581453_c672ba606ceae449ae6e807a9acc4c89.log"
    # u = URLSampleFeatures(url, log_file)
    # u_features = u.extract_url_features()
    # print('')
    d = MySQLReader()
    d.__int__(database="WPF")
    data = d.exec("select * from visit_url_features;")
    w = WebFingerprintSample(data)
    print(w.data)


