from Modules.preprocessing import FPFData, url_list_reader
from Modules.FeatureMining import URLSampleFeatures, flatten_features
import time
import os

## visit_url data
def generate_url_list(datadir, urllst):
    '''
    创建与WPF数据库中visit_urls表结构相同的csv文件，用于更新数据。
    :param datadir: 数据目录顶层路径
    :param urllst: 输出的csv文件路径
    :return: None
    '''
    # datadir = r"D:\projects\WebFingerprinting\data\fpf_data"
    # urllst = r"D:\projects\WebFingerprinting\data_temp\urllst.csv"
    fpf = FPFData(datadir)
    url_list_files = fpf._url_list_files
    with open(urllst, "w") as u:
        for f in url_list_files:
            for url in url_list_reader(f):
                u.write("{0},{1},{2}\n".format(url['timestamp'],
                                                url['url'], url['file_full_path']).replace("\\", "\\\\"))

def generate_url_feature_dataset(dataset, url_list):
    '''
    创建与WPF数据库中visit_url_features表结构相同的csv文件，用于更新数据
    :param dataset: 输出数据路径
    :param url_list: 存有url列表的csv文件
    :return: None
    '''
    writer = open(dataset, "w")
    # names = ['tcp_number', 'main_server_0', 'main_server_1', 'main_server_2', 'main_server_3', 'main_server_4', 'main_common_0', 'main_common_1', 'main_common_2', 'main_common_3', 'main_common_4', 'main_cipher_0', 'main_cipher_1', 'main_cipher_2', 'main_cipher_3', 'main_cipher_4', 'packet_number_0', 'packet_number_1', 'packet_number_2', 'packet_number_3', 'packet_number_4', 'packet_number_5', 'packet_number_6', 'packet_number_7', 'packet_number_8', 'packet_number_9', 'packet_number_10', 'packet_number_11', 'packet_number_12', 'packet_number_13', 'packet_number_14', 'packet_number_15', 'packet_number_16', 'packet_number_17', 'packet_number_18', 'packet_number_19', 'ssl_number_0', 'ssl_number_1', 'ssl_number_2', 'ssl_number_3', 'ssl_number_4', 'ssl_number_5', 'ssl_number_6', 'ssl_number_7', 'ssl_number_8', 'ssl_number_9', 'ssl_number_10', 'ssl_number_11', 'ssl_number_12', 'ssl_number_13', 'ssl_number_14', 'ssl_number_15', 'ssl_number_16', 'ssl_number_17', 'ssl_number_18', 'ssl_number_19', 'min_ssl_size_0', 'min_ssl_size_1', 'min_ssl_size_2', 'min_ssl_size_3', 'min_ssl_size_4', 'min_ssl_size_5', 'min_ssl_size_6', 'min_ssl_size_7', 'min_ssl_size_8', 'min_ssl_size_9', 'min_ssl_size_10', 'min_ssl_size_11', 'min_ssl_size_12', 'min_ssl_size_13', 'min_ssl_size_14', 'min_ssl_size_15', 'min_ssl_size_16', 'min_ssl_size_17', 'min_ssl_size_18', 'min_ssl_size_19', 'max_ssl_size_0', 'max_ssl_size_1', 'max_ssl_size_2', 'max_ssl_size_3', 'max_ssl_size_4', 'max_ssl_size_5', 'max_ssl_size_6', 'max_ssl_size_7', 'max_ssl_size_8', 'max_ssl_size_9', 'max_ssl_size_10', 'max_ssl_size_11', 'max_ssl_size_12', 'max_ssl_size_13', 'max_ssl_size_14', 'max_ssl_size_15', 'max_ssl_size_16', 'max_ssl_size_17', 'max_ssl_size_18', 'max_ssl_size_19', 'mean_ssl_size_0', 'mean_ssl_size_1', 'mean_ssl_size_2', 'mean_ssl_size_3', 'mean_ssl_size_4', 'mean_ssl_size_5', 'mean_ssl_size_6', 'mean_ssl_size_7', 'mean_ssl_size_8', 'mean_ssl_size_9', 'mean_ssl_size_10', 'mean_ssl_size_11', 'mean_ssl_size_12', 'mean_ssl_size_13', 'mean_ssl_size_14', 'mean_ssl_size_15', 'mean_ssl_size_16', 'mean_ssl_size_17', 'mean_ssl_size_18', 'mean_ssl_size_19', 'median_ssl_size_0', 'median_ssl_size_1', 'median_ssl_size_2', 'median_ssl_size_3', 'median_ssl_size_4', 'median_ssl_size_5', 'median_ssl_size_6', 'median_ssl_size_7', 'median_ssl_size_8', 'median_ssl_size_9', 'median_ssl_size_10', 'median_ssl_size_11', 'median_ssl_size_12', 'median_ssl_size_13', 'median_ssl_size_14', 'median_ssl_size_15', 'median_ssl_size_16', 'median_ssl_size_17', 'median_ssl_size_18', 'median_ssl_size_19', 'mode_ssl_size_0', 'mode_ssl_size_1', 'mode_ssl_size_2', 'mode_ssl_size_3', 'mode_ssl_size_4', 'mode_ssl_size_5', 'mode_ssl_size_6', 'mode_ssl_size_7', 'mode_ssl_size_8', 'mode_ssl_size_9', 'mode_ssl_size_10', 'mode_ssl_size_11', 'mode_ssl_size_12', 'mode_ssl_size_13', 'mode_ssl_size_14', 'mode_ssl_size_15', 'mode_ssl_size_16', 'mode_ssl_size_17', 'mode_ssl_size_18', 'mode_ssl_size_19', 'var_ssl_size_0', 'var_ssl_size_1', 'var_ssl_size_2', 'var_ssl_size_3', 'var_ssl_size_4', 'var_ssl_size_5', 'var_ssl_size_6', 'var_ssl_size_7', 'var_ssl_size_8', 'var_ssl_size_9', 'var_ssl_size_10', 'var_ssl_size_11', 'var_ssl_size_12', 'var_ssl_size_13', 'var_ssl_size_14', 'var_ssl_size_15', 'var_ssl_size_16', 'var_ssl_size_17', 'var_ssl_size_18', 'var_ssl_size_19', 'perc_25_0', 'perc_25_1', 'perc_25_2', 'perc_25_3', 'perc_25_4', 'perc_25_5', 'perc_25_6', 'perc_25_7', 'perc_25_8', 'perc_25_9', 'perc_25_10', 'perc_25_11', 'perc_25_12', 'perc_25_13', 'perc_25_14', 'perc_25_15', 'perc_25_16', 'perc_25_17', 'perc_25_18', 'perc_25_19', 'perc_75_0', 'perc_75_1', 'perc_75_2', 'perc_75_3', 'perc_75_4', 'perc_75_5', 'perc_75_6', 'perc_75_7', 'perc_75_8', 'perc_75_9', 'perc_75_10', 'perc_75_11', 'perc_75_12', 'perc_75_13', 'perc_75_14', 'perc_75_15', 'perc_75_16', 'perc_75_17', 'perc_75_18', 'perc_75_19']
    # writer.write(",".join(names))
    # writer.write("\n")
    with open(url_list) as reader:
        for line in reader:
            timestamp, url, log_file = line.rstrip("\n").split(",")
            u = URLSampleFeatures(url, log_file)
            features = u.extract_url_features()
            g = flatten_features(features)
            g.extend([timestamp, url])
            writer.write(",".join(list(map(str, g))))
            writer.write("\n")
            # print(url)

if __name__ == "__main__":
    # generate_url_list(r"D:\BoYun Company\Https_Project\data\8.16\raw_data",
    #                  r"D:\BoYun Company\Https_Project\data\8.16\raw_data\url_list.csv")
    generate_url_feature_dataset(r"D:\BoYun Company\Https_Project\data\8.16\raw_data\dataset_fpf_data_20180816.csv",
                                  r"D:\BoYun Company\Https_Project\data\8.16\raw_data\url_list.csv")