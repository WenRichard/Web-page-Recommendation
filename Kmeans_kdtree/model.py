import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.externals import joblib
from envalue import model_envalue
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN


class kmeans_kdtree_models(object):
    '''
    :param x_train:训练集(特征)
    :param y_trian:训练集(标签)
    :param x_test:测试集(特征)
    :param y_test:测试集(特征)
    :param module_store_path:训练好的kmeans模型的存储路径
    :param file_store_path:训练好的带cluster的数据，往后存库

    算法流程：
    1.function：find_k()：根据聚类的结果图，找到合适的k值，现默认为15
    2.function：train(w)：用kmeans训练数据，得到module和带类标的csv数据
    3.function：predict(w)：用module预测数据，得到cluster，并在cluster内用kd-tree做一个knn计算
                           得到top10个邻居。用建立的评价函数对推荐做出打分
    '''

    def __init__(self,x_train,y_train,x_test,y_test,module_store_path,file_store_path):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.m_path = module_store_path
        self.f_path = file_store_path


    def find_k(self):
        '''
        用tsne降维作图，降维过程耗时比较长，将降维后的数据存成h5文件；根据聚类的结果图，找到合适的k值，现默认为15
        :return:
        '''
        x_train = pd.DataFrame(self.x_train)
        y_train = pd.DataFrame(self.y_train, columns=['url'])
        # tsne = TSNE()
        # x_dsne = tsne.fit_transform(x_train)
        # h5 = pd.HDFStore(r'D:\BoYun Company\Https_Project\data\8.28\car.h5', 'w', complevel=4, complib='blosc')
        # h5['data'] = pd.DataFrame(x_dsne)
        # h5.close()
        x_dsne = np.array(pd.read_hdf(r'D:\BoYun Company\Https_Project\data\8.28\car.h5', key='data'))
        plt.figure(figsize=(12,8))
        for i in range(2,10):
            k = KMeans(n_clusters=i,max_iter=1000, random_state=1).fit_predict(x_dsne)
            colors_lable = ['red','blue','black','yellow','green','brown','maroon','orange','azure','violet', 'rosybrown', 'royalblue', 'pink','mediumblue','plum','seashell','slateblue','wheat']
            colors = ([colors_lable[i] for i in k])
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            #第一次i=2,239+2=241,即2行4列，编号1
            plt.subplot(239+i)
            plt.scatter(x_dsne[:,0],x_dsne[:,1],c=colors,s=10)
            plt.title('K-means Resul of {}'.format(str(i)))
        plt.show()


    def train(self,w):
        '''
        :param w:分为2种情况，if w =='a',将训练数据标准化处理,if w=='b'，不做标准化处理
        :return:
        用kmeans训练数据，得到module和带类标的csv数据
        '''
        x_train = pd.DataFrame(self.x_train)
        y_train = pd.DataFrame(self.y_train,columns=['url'])
        if w =='a':
            sc = StandardScaler()
            scaler = sc.fit(x_train)
            joblib.dump(scaler, r'D:\BoYun Company\Https_Project\data\8.31\8.31fit.pkl')
            x_train = pd.DataFrame(sc.transform(x_train))
        elif w == 'b':
            x_train = x_train
        num_clusters = 15
        kmeans = KMeans(n_clusters=num_clusters, random_state=1)
        kmeans.fit(x_train)
        # 保存模型
        joblib.dump(kmeans, self.m_path)

        # 聚类结果标签
        x_train['cluster'] = kmeans.labels_
        final_data = pd.concat([x_train, y_train], axis=1)
        fd = final_data.to_csv(self.f_path, header=True, index=False)

        # 聚类中心
        centers = kmeans.cluster_centers_

    def predict(self,w):
        '''
        :param  w:分为2种情况，if w =='a',将预测数据标准化处理,if w=='b'，不做标准化处理
        :param: k_lable:聚类后的cluster类标
        :param  y_lable:test数据的lable,[]
        :param  predict:预测出来最相似的10个url,[[],...]
        :return:

        用module预测数据，得到cluster，并在cluster内用kd-tree做一个knn计算
        得到top10个邻居。用建立的评价函数对推荐做出打分
        '''
        kmeans_data = pd.read_csv(self.f_path)
        x_test = pd.DataFrame(self.x_test)
        if w == 'a':
            sc = joblib.load(r'D:\BoYun Company\Https_Project\data\8.31\8.31fit.pkl')
            x_test = pd.DataFrame(sc.transform(x_test))
        elif w == 'b':
            x_test = x_test
        y_lable = self.y_test
        kmeans = joblib.load(self.m_path)
        k_lable = kmeans.predict(x_test)
        predict = []
        for loc,k in enumerate(k_lable):
            #取出在cluster中的数据
            lables_data = kmeans_data[kmeans_data.cluster.isin([k])]
            drop_list = ["cluster", "url"]
            X_ = lables_data.drop(drop_list, axis=1)
            y_ = lables_data.url

            kdt = KDTree(X_, leaf_size=30, metric='euclidean')
            x_query = self.x_test[loc].reshape(1,-1)
            dist, inds = kdt.query(x_query, k=5, return_distance=True)
            dist = dist.squeeze()
            inds = inds.squeeze()
            predict_in=[]
            for r, d in zip(inds, dist):
                predict_in.append(y_.iloc[r])
            predict.append(predict_in)
        score = model_envalue(y_lable,predict)
        print('测试数据的类标为：{}'.format(self.y_test))
        print('预测的top10的类标为：{}'.format(predict))

    def train_predict(self,w):
        self.train(w)
        self.predict(w)

