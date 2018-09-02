
from data_helper import data_preprocess
from data_helper import QD_data
from sklearn.externals import joblib
from model import kmeans_kdtree_models
import time

class models_train(object):
    '''
    :param module_store_path='kmeans.pkl'
    :param file_store_path='kmeans_data.csv'
    '''
    #将没有经过标准化处理的数据集作为输入数据
    def train1(self):
        time1 = time.time()
        x_train,y_train = data_preprocess().SSL_train_transform()
        x_test, y_test = data_preprocess().SSL_test_transform()
        module_store_path = 'D:\BoYun Company\Https_Project\module result\output\kmeans\8.31 raw\8.31carkmeans.pkl'
        file_store_path = 'D:\BoYun Company\Https_Project\module result\output\kmeans\8.31 raw\8.31carkmeans_data.csv'
        kmeans_kdtree_models(x_train,y_train,x_test,y_test,module_store_path,file_store_path).train_predict('b')
        time2 = time.time()
        print('module cost:{}'.format((time2 - time1)))

    #利用起点中文网的数据做实验
    def train2(self):
        time1 = time.time()
        x_train,y_train = QD_data().QD_train()
        x_test, y_test = QD_data().QD_test_()
        module_store_path = r'D:\BoYun Company\Https_Project\module result\output\kmeans\8.30 raw\8.30bookkmeans.pkl'
        file_store_path = r'D:\BoYun Company\Https_Project\module result\output\kmeans\8.30 raw\8.30bookkmeans_data.csv'
        kmeans_kdtree_models(x_train,y_train,x_test,y_test,module_store_path,file_store_path).train_predict('b')
        time2 = time.time()
        print('module cost:{}'.format((time2-time1)))

    #查看聚类效果，找到合适的k
    def train3(self):
        x_train, y_train = data_preprocess().SSL_train_transform()
        x_test, y_test = 1,1
        module_store_path = r'D:\BoYun Company\Https_Project\data\8.27\book_test\kmeans.pkl'
        file_store_path = r'D:\BoYun Company\Https_Project\data\8.27\book_test\kmeans_data.csv'
        kmeans_kdtree_models(x_train,y_train,x_test,y_test,module_store_path,file_store_path).find_k()

    #将经过标准化处理的数据集作为输入数据
    def train4(self):
        x_train,y_train = data_preprocess().SSL_train_transform()
        x_test, y_test = data_preprocess().SSL_test_transform()
        module_store_path = r'D:\BoYun Company\Https_Project\data\8.28\kmeans.pkl'
        file_store_path = r'D:\BoYun Company\Https_Project\data\8.28\kmeans_data.csv'
        kmeans_kdtree_models(x_train,y_train,x_test,y_test,module_store_path,file_store_path).train_predict('a')

if __name__ == '__main__':
    #models_train().train()
    #models_train().train3()
    models_train().train1()