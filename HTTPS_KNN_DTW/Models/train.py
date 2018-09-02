from Models.K_D_model import KnnDtw
from Models.data_helper import data_preprocess
from Models.envalue import model_envalue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle

class model_train(object):
    '''
    :param step:决定测试集有多少
    :param win:window size的大小

    '''
    def __init__(self,step=1,win = 1):
        self.time = np.linspace(0, 20, 1000)
        self.amplitude_a = 5 * np.sin(self.time)
        self.amplitude_b = 3 * np.sin(self.time + 1)
        self.step = step
        self.win = win

    def sample_test(self):
        m = KnnDtw()
        distance = m._dtw_distance(self.amplitude_a, self.amplitude_b)
        fig = plt.figure(figsize=(12, 4))
        plt.plot(self.time, self.amplitude_a, label='A')
        plt.plot(self.time, self.amplitude_b, label='B')
        plt.title('DTW distance between A and B is %.2f' % distance)
        plt.ylabel('Amplitude')
        plt.xlabel('Time')
        plt.legend()
        plt.show()
        print(m._dist_matrix(np.random.random((4, 50)), np.random.random((4, 50))))

    #获取最为合适的windows size
    def envalue_window_size(self):
        data = data_preprocess().SSL_data()
        x_train = data[0]
        y_train = data[1]
        x_test = data[2]
        y_test = data[3]
        time_taken = []
        windows = [1, 2, 5, 10, 50, 100, 500, 1000, 5000]
        for w in windows:
            begin = time.time()
            t = KnnDtw(n_neighbors=1, max_warping_window=w)
            t.fit(x_train, y_train)
            label, proba = t.predict(x_test[:20])

            end = time.time()
            time_taken.append(end - begin)
        fig = plt.figure(figsize=(12, 5))
        _ = plt.plot(windows, [t / 400. for t in time_taken], lw=4)
        plt.title('DTW Execution Time with \nvarying Max Warping Window')
        plt.ylabel('Execution Time (seconds)')
        plt.xlabel('Max Warping Window')
        plt.xscale('log')
        plt.show()

    def SSL_test(self):
        f = open('index_lable_dic.pickle', 'rb')
        dic = pickle.load(f)
        train_data = data_preprocess().SSL_train_transform('a')
        test_data = data_preprocess().SSL_test_transform('a')
        x_train = train_data[0]
        y_train = train_data[1]
        x_test = test_data[0]
        self.y_test = test_data[1]
        self.labels = test_data[2]
        time1 = time.time()
        m = KnnDtw(n_neighbors=10, max_warping_window=self.win)
        #m.fit(x_train[::10], y_train[::10])
        m.fit(x_train, y_train)

        #self.label, self.proba = m.predict(x_test[::self.step])
        self.label = m.predict(x_test[::self.step])

        time2 = time.time()

        print('DTW的step:{}'.format(self.step))
        print('DTW的window size：{}'.format(self.win))
        print('DTW cost time:{} s'.format(time2 - time1))
        #print(self.proba,file=f)
        model_envalue(self.y_test[::self.step],self.label)
        print(self.y_test[::self.step])
        print(self.label)


    # 将包含索引的列表转换成url的列表
    def transformer(self, input_index, output_index):
        f = open('index_lable_dic.pickle', 'rb')
        dic = pickle.load(f)
        self.input_lable = [dic[r] for r in input_index]
        self.output_lable = []
        for t1 in output_index:
            t2 = [dic[r] for r in t1]
            self.output_lable.append(t2)
        return self.input_lable, self.output_lable

if __name__ == '__main__':
    model_train().SSL_test()