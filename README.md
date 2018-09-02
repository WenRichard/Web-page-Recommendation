# Web-page-Recommendation
***option*** :Use SSL sequences for page recommendations
## Data Description 
Data中包含两种网页类型的数据。<br>
**静态网页**：起点中文网<br>
1.	Qidian.com <br>
Train：19015   100种url <br>
Test： 1900    96种url <br>

**动态网页**：汽车之家<br>
2.	autohome.com <br>
Train：28316   140种url  <br>
Test： 771    20种url    <br>
## Run Modules 
```
一.文件夹有4个模型
--h
--Raw_data_Modules
--HTTPS_KNN_DTW
--Kmeans_kdtree
--Sax_knn

1.Raw_data_Modules
仅供模型生成ssl序列数据使用。
使用方法：调用 generate_ssl_seq2(path1='',path2='')即可生成h5文件

2.HTTPS_KNN_DTW
利用处理时间序列比较流行的DTW算法进行实验。
缺点：时间复杂度很高，几乎接近o（n*3）--n的3次方
介绍及使用方法：
--h
--data_help.py
--envalue.py
--K_D_model.py
--train.py

--data_help.py
需要设置的几个参数：
w：w为设定的ssl序列长度取值
t：t为设定的测试url种类取值
data_path1: data_path1为读取的ssl序列csv文件或者h5文件，训练数据
data_path2: data_path2为读取的ssl序列csv文件或者h5文件，测试数据
------
如果在新环境中，先运行data_preprocess().SSL_dic()，生成需要的2个字典

--K_D_model.py
需要设置的几个参数：
self.n_neighbors ：knn的n，最近的n个邻居
self.max_warping_window ：dtw优化的窗口
self.subsample_step ：dtw寻找前进的step，默认为1
使用方法：
m = KnnDtw(n_neighbors=n, max_warping_window=m)
m.fit(x_train, y_train)
self.label = m.predict(x_test[::self.step])

--envalue.py
需要设置的几个参数：
lables: 1-d array：exp:[29 29 29 29 29 29 29 29]
predict: 2-d array：exp:[[29 29 29 29 29 29 29 29]，[23, 105, 81, 97, 75, 74, 42, 132, 54, 4]]
record_path:生成的结果地址
envalue_path:生成的总评价地址
使用方法：
model_envalue(lables,predict)

--train.py
需要设置的几个参数：
step: 决定测试集有多少
win: window size的大小
使用方法：
model_train().SSL_test()

3.Kmeans_kdtree
先聚类再用knn寻找最近的topk个url作为推荐url
效率：效果较好，时间复杂度较低，在现有数据上基本1s内出结果
数据没做标准化之前，k=10，标准化之后，k=5，因为标准化后，没有那么多的维度可以划分，会报错。
介绍及使用方法：
--h
--data_help.py
--envalue.py
--model.py
--train.py

--data_help.py
需要设置的几个参数：
w：w为设定的ssl序列长度取值
t：t为设定的测试url种类取值
这个文件中有2个数据源，汽车之家和起点中文网
------
如果在新环境中，先运行data_preprocess().SSL_dic()，生成需要的2个字典

--envalue.py
需要设置的几个参数：
lables: 1-d array：exp:[29 29 29 29 29 29 29 29]
predict: 2-d array：exp:[[29 29 29 29 29 29 29 29]，[23, 105, 81, 97, 75, 74, 42, 132, 54, 4]]
使用方法：
model_envalue(lables,predict)

--model.py
需要设置的几个参数：
x_train：训练集(特征)
y_train：训练集(标签)
x_test：测试集(标签)
y_test：测试集(特征)
module_store_path：训练好的kmeans模型的存储路径
file_store_path：训练好的带cluster的数据路径，往后存库
使用方法：
kmeans_kdtree_models(x_train,y_train,x_test,y_test,module_store_path,file_store_path).train_predict('b')

--train.py
需要设置的几个参数：
step: 决定测试集有多少
win: window size的大小
使用方法：
models_train().train1()

需要注意的点：
读取数据过程，数据很有可能存在空值的情况，目前已经解决，但是出错时，也请往这上面考虑。
注意数据路径，算法有多次读取数据生成字典的过程，不同的数据或生成不同的字典。

```
