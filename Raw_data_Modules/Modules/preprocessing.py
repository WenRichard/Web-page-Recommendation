import numpy as np
import pandas as pd
import csv
import os


# import torch
# from torch.utils.data import Dataset, DataLoader

class WebPageNumpy(object):
    def __init__(self, npz_file="", website_No_file=""):
        self.npz_file = npz_file
        t = np.load(self.npz_file)
        self.data = t['data']
        labels = t['labels']
        websites_No = pd.read_table(website_No_file)
        websites = list(websites_No['name'])
        No = list(websites_No['No'])
        labels_dict = {}
        for w, N in zip(websites, No):
            labels_dict[str(w)] = N
        for i in range(len(labels)):
            labels[i] = labels_dict[str(labels[i])]
        self.labels = labels.astype(np.int32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        d = self.data[idx]
        l = self.labels[idx]
        return d, l

    def get_data(self):
        return self.data, self.labels


class FPFData(object):
    '''
    Fingerprint Features.
    根据结构化的指纹数据存储目录，实现数据的读取和迭代
    '''
    def __init__(self, data_dir):
        assert os.path.exists(data_dir) is True
        self._data_dir = data_dir
        self._url_list_files = [a for a in self.walk_dir() if a.endswith(".lst")]

    def get_sites_list(self):
        '''当前数据目录下的所有网站，写入到临时文本文件中'''
        self.sites_list = os.listdir(self._data_dir)
        return self.sites_list

    def walk_dir(self):
        '''遍历目录下的子目录和文件，得到所有文件的路径'''
        for root, dirs, files in os.walk(self._data_dir):
            for f in files:
                abs_file_path = os.path.join(root, f)
                yield abs_file_path

    def walk_logs(self):
        '''遍历所有文件并得到所有log文件的路径'''
        log_files = []
        for one in self.walk_dir():
            if one.endswith(".log"):
                log_files.append(one)
        return log_files
    def walk_url_lst(self, url_list_file="./data_temp/url_list.txt"):
        if os.path.exists(url_list_file):
            os.remove(url_list_file)
        with open(url_list_file, "w+") as urls:
            for one in self.walk_dir():
                if one.endswith(".lst"):
                    with open(one, "r") as url_lst:
                        for line in url_lst:
                            time_stamp, url, log_file = line.rstrip("\n").split(",")
                            log_file = os.path.join(os.path.split(one)[0], log_file)
                            urls.write("{0},{1},{2}\n".format(time_stamp, url, log_file))

                else:
                    pass


    def fetch_one_iter(self, log_file_path="None"):
        '''
        指纹数据生成器
        创建一个返回单条记录生成器，每次返回一个url和指纹特征字典
        '''
        print(log_file_path)
        if log_file_path == "None":
            for log_file in self._log_files:
                path, name = os.path.split(log_file)
                one = URLSample(log_file)
                url = [u for u in url_list_reader(os.path.join(path, "url.lst")) \
                       if u['file_name'] == name][0]
                yield url, one.TCP_feature_list
        else:
            path, name = os.path.split(log_file_path)
            url = [u for u in url_list_reader(os.path.join(path, "url.lst")) \
                   if u['file_name'] == name][0]
            one = URLSample(log_file_path)
            return url, one.TCP_feature_list

    def __len__(self):
        return len(self._url_list_files)

    def __getitem__(self, item):
        fetcher = self.fetch_one_iter()
        i = 0
        for j in fetcher:
            if i == item:
                return j
            else:
                i += 1


class URLSample(object):
    '''
    对url指纹特征.log文件进行处理，提取需要的特征，并格式化为一个样本
    '''

    def __init__(self, log_file):
        assert os.path.exists(log_file)
        self.log_file = log_file
        self.keys = (
            'URL',
            'NO',
            'src and dst addr',
            'server name',
            'Cipher Suite',
            'common name',
            'packet number',
            'ssl load number',
            'ssl load message',
            'object message',
            'original message',
        )
        self.fake_features = {
            'URL':"Unknown",
            'NO': 0,
            'src and dst addr': "0.0.0.0:0<--->0.0.0.0:0",
            'server name': "Unknown",
            'Cipher Suite': 'xxx UnknownCipherSuite Block Size:0 Hash Size:0',
            'common name': "Unknown",
            'packet number': "0",
            'ssl load number': "0",
            'ssl load message': "0 0 0",
            'object message': "( 0 , 0 )",
            'original message': "( 0 , 0 )",
        }
        self.TCP_feature_list = self.get_TCP_feature_list()

    def get_TCP_feature_list(self):
        with open(self.log_file, "r") as f:
            g = f.read().replace("\n", "")
            gg = eval("[%s]" % g)
            for i in range(len(gg)):
                for k in self.keys:
                    if k not in gg[i].keys():
                        gg[i][k] = self.fake_features[k]
            tcp_feature_list = list(map(TCPFeatures, gg))
            return tcp_feature_list
            # x = list(map(lambda d: d.values(), gg))
            # return x
            # return pd.DataFrame(x, columns=self.keys)

class TCPFeatures(object):
    def __init__(self, tcp_dict):
        self.tcp_dict = tcp_dict
        self.features = self.extract_features()
    def __getitem__(self, key):
        return self.features[key]
    def __setitem__(self, key, value):
        self.features[key] = value
    def keys(self):
        return self.features.keys()
    def extract_features(self):
        d = self.tcp_dict
        features = {}
        No = int(d['NO'][12:])
        features['NO'] = No
        src_dst = d['src and dst addr']
        t = list(map(lambda a: a.split(":"), src_dst.split("<--->")))
        # src_ip, src_port = t[0]
        # dst_ip, dst_port = t[1]
        features['src_ip'], features['src_port'] = t[0]
        features['dst_ip'], features['dst_port'] = t[1]
        features['server_name'] = d['server name']
        features['common_name'] = d['common name']
        features['cipher_suite'] = d['Cipher Suite'].split(" ")[1]
        features['packet_number'] = int(d['packet number'])
        features['ssl_load_number'] = int(d['ssl load number'])
        features['ssl_load_message'] = list(map(int, d['ssl load message'].split(" ")[:-1]))
        features['object_message'] = eval(d['object message'])
        return features


def url_list_reader(url_lst_path):
    log_dir = os.path.split(url_lst_path)[0]
    with open(url_lst_path, "r") as f:
        lines = csv.reader(f)
        for line in lines:
            yield {"timestamp": line[0],
                   "url": line[1],
                   "file_name": line[2],
                   "file_full_path": os.path.join(log_dir, line[2])}

#如果backup_path = "a",则代表不会生成h5文件
def generate_ssl_seq(data_path = r"D:\BoYun Company\Https_Project\data\8.20\www.autohome.com.cn",
                     backup_path = r"D:\BoYun Company\Https_Project\data\8.20\www.autohome.com.cn\8.20_car_ssl_seq.h5"):
    fpf = FPFData(data_path)
    fpf.walk_url_lst("urllist.txt")
    url_list = pd.read_table(r"urllist.txt", header=None).values.tolist()
    url_ssl_seq = []
    for r in url_list:
        url = r[0].split(',')[1]
        log_file_path = r[0].split(',')[2]
        u = URLSample(log_file=log_file_path)
        gg = u.get_TCP_feature_list()
        ssl_seq = []
        rest_tcp = 10 - len(gg)
        if rest_tcp > 0:
            #len(gg)代表一个url里tcp的个数，现在默认设置tcp个数为10，如果原始数据不够10的话，会生成假数据进行填充
            for i in range(len(gg)):
                g = gg[i]
                # for k in g.keys():
                #     print(k, ":", g[k])
                ssl_list = g['ssl_load_message']
                rest_len = 200 - len(ssl_list)
                if rest_len > 0:
                    rest_0 = np.zeros(rest_len, dtype=int).tolist()
                    ssl_list.extend(rest_0)
                else:
                    ssl_list = [ssl_list[:200]]
                ssl_seq.extend(ssl_list)
            #如果tcp数不足10，再构造剩下的伪数据
            for i in range(rest_tcp):
                ssl_seq.extend(np.zeros(200,dtype=int).tolist())
        else:
            for i in range(10):
                g = gg[i]
                ssl_list = g['ssl_load_message']
                rest_len = 200 - len(ssl_list)
                if rest_len > 0:
                    rest_0 = np.zeros(rest_len, dtype=int).tolist()
                    ssl_list.extend(rest_0)
                else:
                    ssl_list = [ssl_list[:200]]
                ssl_seq.extend(ssl_list)
        ssl_seq.append(url)
        url_ssl_seq.append(ssl_seq)
    pd_seq = pd.DataFrame(url_ssl_seq)
    print(pd_seq.shape)
    if backup_path == 'a':
        pass
    else:
        h5 = pd.HDFStore(backup_path, 'w', complevel=4, complib='blosc')
        h5['data'] = pd_seq
        h5.close()
    return pd_seq

def generate_ssl_seq2(data_path = r"D:\BoYun Company\Https_Project\data\8.27\car_test",
                     backup_path = r"D:\BoYun Company\Https_Project\data\8.27\car_test\8.27_car_test.h5"):
    fpf = FPFData(data_path)
    fpf.walk_url_lst("urllist.txt")
    url_list = pd.read_table(r"urllist.txt", header=None).values.tolist()
    url_ssl_seq = []
    for r in url_list:
        url = r[0].split(',')[1]
        log_file_path = r[0].split(',')[2]
        u = URLSample(log_file=log_file_path)
        gg = u.get_TCP_feature_list()
        ssl_seq = []
        rest_tcp = 10 - len(gg)
        if rest_tcp > 0:
            #len(gg)代表一个url里tcp的个数，现在默认设置tcp个数为10，如果原始数据不够10的话，会生成假数据进行填充
            for i in range(len(gg)):
                g = gg[i]
                # for k in g.keys():
                #     print(k, ":", g[k])
                ssl_list = g['ssl_load_message']
                rest_len = 200 - len(ssl_list)
                if rest_len > 0:
                    rest_0 = np.zeros(rest_len, dtype=int).tolist()
                    ssl_list.extend(rest_0)
                else:
                    ssl_list = [ssl_list[:200]]
                ssl_seq.extend(ssl_list)
            #如果tcp数不足10，再构造剩下的伪数据
            for i in range(rest_tcp):
                ssl_seq.extend(np.zeros(200,dtype=int).tolist())
        else:
            for i in range(10):
                g = gg[i]
                ssl_list = g['ssl_load_message']
                rest_len = 200 - len(ssl_list)
                if rest_len > 0:
                    rest_0 = np.zeros(rest_len, dtype=int).tolist()
                    ssl_list.extend(rest_0)
                else:
                    ssl_list = [ssl_list[:200]]
                ssl_seq.extend(ssl_list)
        ssl_seq.append(url)
        url_ssl_seq.append(ssl_seq)
    pd_seq = pd.DataFrame(url_ssl_seq)
    print(pd_seq.shape)
    if backup_path == 'a':
        pass
    else:
        h5 = pd.HDFStore(backup_path, 'w', complevel=4, complib='blosc')
        h5['data'] = pd_seq
        h5.close()
    return pd_seq










if __name__ == "__main__":
    generate_ssl_seq2()

    # for k in g.keys():
    #      print(k, ":", g[k])
    #fpf = FPFData(r"D:\BoYun Company\Work File\fpf_data\fpf_data")
    #print("hah")




