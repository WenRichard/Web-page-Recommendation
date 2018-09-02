import pickle
import pandas as pd

#envalue principle：
def model_envalue(lables,predict):
    '''
    :param lables: 1-d array：exp:[29 29 29 29 29 29 29 29]
    :param predict: 2-d array：exp:[[29 29 29 29 29 29 29 29]，[23, 105, 81, 97, 75, 74, 42, 132, 54, 4]]

    envalue principle：

    1.return：
    for every predict record,it will be shown as numbers of m1,m3,m4,m5,m6 and the number of precise
    recommend url,the recommend score above all

    2.return:
    generate a DataFrame where clounmns are 'lable','url','precise counts' ,'ave score'
    lable: url's index
    url : url
    precise counts：sum of precise recommend url's counts per url class
    ave score：sum of recommend score's counts/ url's len

    精准推荐个数评判：保证至少每条记录推荐出来的10个url能够有1条是精准推荐的，假如有n条记录，至少得有n条精准推荐，这样的话
    至少保证了推荐出来的url至少有一条是正确的。
    '''

    f = open(r'index_lable_dic.pickle', 'rb')
    dic = pickle.load(f)
    input_index, output_index = lables, predict

    lable_count = {}
    for i,j in enumerate(input_index):
        if j not in lable_count:
            lable_count[j] = 1
        else:
            lable_count[j] +=1

    #随机选取的url经过pd isin后会进行类之间的排序，比如url1后面会是n个url1(不会出现url1，url2的情况)
    forward = input_index[0]
    lable_len = 0
    score_1 = 0
    m_1_count = 0
    ave_s = []
    for i,real in enumerate(input_index):
        w1 = 1
        w2 = 0
        w3 = 0.8
        w4 = 0.1
        w5 = 0.2
        w6 = 0.1
        real = real
        f = open('D:\BoYun Company\Https_Project\module result\output\kmeans\8.31 raw\8.31carresult.txt', 'a')
        print('---------------------------------------------------START-----------------------------------------------',
              file=f)
        print('输入的url为：{}'.format(dic[real]),file=f)
        print('输入的url为：{}'.format(dic[real]))
        pred = output_index[i]
        print('预测的url个数为：{}'.format(len(pred)),file=f)
        print('预测的url个数为：{}'.format(len(pred)))

        print('--------------------------------预测的top10个URL------------------------------------', file=f)
        print('--------------------------------预测的top10个URL------------------------------------')
        for t in pred:
            print('预测的url分别为{},{}'.format(t, dic[t]), file=f)
            print('预测的url分别为{},{}'.format(t, dic[t]))

        m1 = 0
        m3 = 0
        m4 = 0
        m5 = 0
        m6 = 0
        for w in pred:
            # 判断url的索引是否一致
            if real == w :
                m1 += 1
            else:
                # 判断split后url域名后’/’的第一个元素是否相等
                split_1 = lambda x:dic[x].strip().split('//')[1].split('/')
                real_1 = split_1(real)
                #去除列表中的‘’
                if real_1[-1] =='':
                    del real_1[-1]
                pred_1 = split_1(w)
                if pred_1[-1] =='':
                    del pred_1[-1]
                try:
                    real_2 = real_1[1]
                    pred_2 = pred_1[1]
                    if real_2 == pred_2:
                        m3 +=1
                    else:
                        if len(real_1) == len(pred_1):
                            if real_1[-1][0] == pred_1[-1][0]:
                                m5 +=1
                            else:
                                m6 +=1
                        else:
                            m4 +=1
                except:
                    m3 = 0
                    m4 = 0
                    m5 = 0
                    m6 = 0
        print('m1的个数为：{}'.format(m1),file=f)
        print('m3的个数为：{}'.format(m3),file=f)
        print('m4的个数为：{}'.format(m4),file=f)
        print('m5的个数为：{}'.format(m5),file=f)
        print('m6的个数为：{}'.format(m6),file=f)

        print('m1的个数为：{}'.format(m1))
        print('m3的个数为：{}'.format(m3))
        print('m4的个数为：{}'.format(m4))
        print('m5的个数为：{}'.format(m5))
        print('m6的个数为：{}'.format(m6))

        print('精准推荐的url个数为：{}'.format(m1),file=f)
        print('精准推荐的url个数为：{}'.format(m1))

        score = m1*w1+m3*w3+m4*w4+m5*w5+m6*w6
        print('recommend score:{}'.format(score),file=f)
        print('recommend score:{}'.format(score))

        print('--------------------------------------------------------END------------------------------------------',file=f)
        print('--------------------------------------------------------END------------------------------------------')
        if forward == real:
            score_1 = score_1 + score
            m_1_count = m_1_count + m1
            lable_len +=1
        else:
            in_ = []
            ave_score = score_1/lable_len
            #添加lable,url,精准推荐个数，平均分数值
            in_.append(forward)
            in_.append(dic[forward])
            in_.append(m_1_count)
            in_.append(ave_score)
            ave_s.append(in_)
            lable_len = 1
            forward = real
            score_1 = score
            m_1_count = m1
        #终止条件,借助字典查询该url的条数
        if i == len(input_index)-1:
            real_count = lable_count[real]
            ave_score = score_1/real_count
            in_2= []
            in_2.append(real)
            in_2.append(dic[real])
            in_2.append(m_1_count)
            in_2.append(ave_score)
            ave_s.append(in_2)
    principle = pd.DataFrame(ave_s,columns=['lable','url','precise counts','ave score'])
    principle.to_csv(r'D:\BoYun Company\Https_Project\module result\output\kmeans\8.31 raw\8.31carprinciple.csv',index=False)


