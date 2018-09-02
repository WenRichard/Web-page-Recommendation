import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import pickle

class data_preprocess(object):
    '''
    参数：
    w：w为设定的ssl序列长度取值
    t：t为设定的测试url种类取值
    '''
    def __init__(self,w=50,t=1):
        self.w = w
        self.t = t

    def SSL_data(self):
        #分割数据集
        ssl_data = pd.read_csv('D:\BoYun Company\Https_Project\data\8.20\www.autohome.com.cn\seq_data_20180820.csv')
        #URL为类标
        url = np.unique(ssl_data.iloc[:,2000].values.tolist()).tolist()
        print('总共有{}种url'.format(len(url)))
        url_dict = {}
        for i,k in enumerate(url):
            url_dict[k] = i
        f = open('lable_index_dic.pickle','wb')
        pickle.dump(url_dict,f)
        f.close()
        #随机选取self.t种url
        slice_ = random.sample(url, self.t)
        lables = {}
        for r in slice_:
            index = url_dict[r]
            lables[index] = r
            print('选取的url分别为{},{}'.format(index,r))
        #选取对应lable的数据
        ch_data = []
        for r2 in slice_:
            choose_data = ssl_data[ssl_data.iloc[:,2000].isin([r2])]
            ch_data.append(choose_data)
        ch_data2 = pd.concat(ch_data)
        print(ch_data2.shape)
        #将选取的数据集分割成训练集，测试集，ratio=0.2
        X = ch_data2.iloc[:,0:2000]
        y = ch_data2.iloc[:,2000]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        #将分割的数据去0变成模型的标准输入
        X_train = np.array(X_train.iloc[:,0:self.w].values.tolist())
        X_test = np.array(X_test.iloc[:, 0:self.w].values.tolist())
        y_train = np.array([url_dict[i] for i in y_train.values.tolist()])
        y_test = np.array([url_dict[i] for i in y_test.values.tolist()])
        print('x_train的shape为：{}'.format(X_train.shape))
        print('y_train的shape为：{}'.format(y_train.shape))
        print('x_test的shape为：{}'.format(X_test.shape))
        print('y_test的shape为：{}'.format(y_test.shape))
        return X_train, y_train, X_test, y_test, lables

    # 将ssl序列的padding的0去掉，暂时不需要
    def del_0(self,data):
        list1 = []
        for r3 in data:
            list2 = []
            for r4 in r3:
                if r4 != 0:
                    list2.append(r4)

            list1.append(list2)
        print(len(list1))
        print(np.array(list1))
        return list1

    # 制作一个lable-index的字典
    # 制作一个index-lable的字典
    def SSL_dic(self):
        ssl_data = pd.read_csv('D:\BoYun Company\Https_Project\data\8.20\www.autohome.com.cn\seq_data_20180820.csv')
        # URL为类标
        url = np.unique(ssl_data.iloc[:, 2000].values.tolist()).tolist()
        print('总共有{}种url'.format(len(url)))
        url_dict = {}
        index_lable = {}
        for i, k in enumerate(url):
            url_dict[k] = i
            index_lable[i] = k
        f = open('lable_index_dic.pickle', 'wb')
        f2 = open('index_lable_dic.pickle', 'wb')
        pickle.dump(url_dict, f)
        pickle.dump(index_lable, f2)
        f.close()

    # 训练集处理，将之前的2000维处理成50维
    def SSL_train_transform(self,t):
        '''
        :param t: if t == 'a':按照ssl原始的序列进行处理
                  if t == 'b':按照ssl降序后的序列进行处理
        :return:
        ssl_data既可以用read_hdf读，也可以用read_csv读，没有影响
        '''
        f = open('lable_index_dic.pickle','rb')
        dic = pickle.load(f)
        f.close()
        #ssl_data = pd.read_hdf(r'D:\BoYun Company\Https_Project\data\8.22\20180821\8.21_car_ssl_seq.h5', key='data')
        ssl_data = pd.read_csv('D:\BoYun Company\Https_Project\data\8.20\www.autohome.com.cn\seq_data_20180820.csv')
        # URL为类标
        url = np.unique(ssl_data.iloc[:, 2000].values.tolist()).tolist()
        print('训练数据总共有{}种url'.format(len(url)))
        
        lables = dic
        if t =='a':
            X_train = np.array(ssl_data.iloc[:, 0:self.w].values.tolist())
        elif t == 'b':
            x_test = ssl_data.iloc[:, 0:self.w].values.tolist()
            X_train = []
            for i in x_test:
                X_train.append(sorted(i,reverse=True))
            X_train=np.array(X_train)
        y_train = np.array([dic[i] for i in ssl_data.iloc[:,2000].values.tolist()])
        return X_train,y_train

    def SSL_test_transform(self,t):
        '''
        :param t: if t == 'a':按照ssl原始的序列进行处理
                  if t == 'b':按照ssl降序后的序列进行处理
        :return:
         ssl_data既可以用read_hdf读，也可以用read_csv读，没有影响
        '''
        f = open('lable_index_dic.pickle','rb')
        dic = pickle.load(f)
        f.close()
        ssl_data = pd.read_hdf(r'D:\BoYun Company\Https_Project\data\8.22\20180821\8.21_car_ssl_seq.h5', key='data')
        # URL为类标
        url = np.unique(ssl_data.iloc[:, 2000].values.tolist()).tolist()
        print('测试数据总共有{}种url'.format(len(url)))
        # 随机选取self.t种url
        slice_ = random.sample(url, self.t)
        lables = {}
        for r in slice_:
            index = dic[r]
            lables[index] = r
            print('选取的url分别为{},{}'.format(index, r))
        # 选取对应lable的数据
        ch_data = []
        for r2 in slice_:
            choose_data = ssl_data[ssl_data.iloc[:, 2000].isin([r2])]
            ch_data.append(choose_data)
        ch_data2 = pd.concat(ch_data)
        print(ch_data2.shape)
        if t == 'a':
            X_test = np.array(ch_data2.iloc[:, 0:self.w].values.tolist())
        elif t == 'b':
            x_test = ssl_data.iloc[:, 0:self.w].values.tolist()
            X_test = []
            for i in x_test:
                X_test.append(sorted(i, reverse=True))
            X_test = np.array(X_test)
        y_test = np.array([dic[i] for i in ch_data2.iloc[:,2000].values.tolist()])
        return X_test,y_test,lables





if __name__ == '__main__':
    #data_preprocess().HAR_data()
    #data_preprocess().SSL_data()
    #data_preprocess().SSL_test_transform()
    data_preprocess().SSL_dic()