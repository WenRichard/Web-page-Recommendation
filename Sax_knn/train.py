from data_helper import data_preprocess
from data_helper import QD_data
from sklearn.externals import joblib
from sax_module import SAX
from envalue import model_envalue
import time


class module_train(object):

    def train1(self):
        time1 = time.time()
        s = SAX()
        x_train, y_train = data_preprocess().SSL_train_transform()
        x_test, y_test = data_preprocess().SSL_test_transform()
        y_lable,y_predict = s.knn_predict(x_train,y_train,x_test,y_test)
        score = model_envalue(y_lable,y_predict)
        time2 =time.time()
        print('Sax_knn cost:{}'.format((time2-time1)))




if __name__ == '__main__':
    module_train().train1()