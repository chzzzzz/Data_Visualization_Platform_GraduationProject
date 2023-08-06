#!/usr/bin/env python
# coding: utf-8

# In[3]:


# In[33]:


import numpy as np
import pandas as pd
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
from tslearn.clustering import silhouette_score
from sklearn.preprocessing import MinMaxScaler  # 调用了scikit-learn对象MinMaxScaler规范化数据集

"""
1.数据读取与预处理（序列填充，使每条序列等长）
2.计算轮廓系数，求出轮廓系数最大时的聚类个数k
3.使用最佳聚类个数，得到序列聚类标签
4.可视化，绘制elbow线图辅助检验聚类个数是否合理，同时绘制不同序列的聚类效果图。
"""


class Plot_Cluster_Time_Series(object):
    def __init__(self, data, seed):
        self.data = data
        self.seed = seed

    # 补全每个日期上的值
    #     def fill_na_ts(self):
    #         data=self.data
    #         #取item_id无重复部分
    #         df_store = data[['item_id']].drop_duplicates()
    #         max_ds = str(data['date'].max())[:10].replace('-', '')
    #         min_ds = str(data['date'].min())[:10].replace('-', '')
    #         print('min time is : {},max time is : {}'.format(min_ds, max_ds))
    #         #补齐每一天
    #         time_index = pd.date_range(min_ds, max_ds, freq='D')
    #         time_index = pd.DataFrame(time_index)
    #         time_index.columns = ['ts_index']
    #         time_index['value'] = 1
    #         df_store['value'] = 1
    #         store_time_index = pd.merge(time_index, df_store, how='left', on='value')
    #         store_time_index.drop(columns='value', inplace=True)
    #         data['date'] = pd.to_datetime(data['date'])
    #         store_time_index['ts_index'] = pd.to_datetime(store_time_index['ts_index'])
    #         store_time_index.rename(columns={'ts_index': 'date'}, inplace=True)
    #         data_full = pd.merge(store_time_index, data, how='left', on=['date', 'item_id'])
    #         data_full['qty'] = data_full['qty'].fillna(0)
    #         data_full.fillna(0, inplace=True)
    #         return data_full
    def read_data(self, data):
        """
        :return: norm dataset and time series id
        """
        s = ["负载电流", "负载电压", "逆变电流", "逆变电压", "输入电流", "输入电压"]
        input = data[s]
        input = input.values
        scaler = MinMaxScaler(feature_range=(0, 1))  # 对数据进行归一化
        scaler = scaler.fit(input)
        input = scaler.transform(input)
        w = np.zeros(shape=(199, 9, 6))
        count = 0
        for i in range(0, 199):
            for j in range(0, 9):
                if (count < len(input)):
                    for k in range(0, 6):
                        w[i][j][k] = input[count][k]
                    count = count + 1
                else:
                    break
        # ts_norm = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(w)
        #         multi_ts = data.sort_values(by=['item_id', 'date'], ascending=[1, 1])[['item_id', 'qty']]
        #         #行数/id不同值的个数
        #         int_numer=multi_ts.shape[0] // multi_ts['item_id'].nunique()
        #         multi_ts=multi_ts.groupby('item_id').filter(lambda x: x['item_id'].count() ==int_numer)
        ts_norm = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(w)
        return ts_norm

    # , multi_ts['item_id'].unique()

    def plot_elbow(self, data):
        """
        :param df:multi time series  type is np.array
        :return: elbow plot
        """
        distortions = []
        for i in range(2, 7):
            ks = KShape(n_clusters=i, n_init=5, verbose=True, random_state=self.seed)
            ks.fit(data)
            distortions.append(ks.inertia_)
        plt.plot(range(2, 7), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion Line')
        plt.show()

    def shape_score(self, data, labels, metric='dtw'):
        """
        :param df:
        :param labels:
        :param metric:
        :return:
        """
        score = silhouette_score(data, labels, metric)
        return score

    def cal_k_shape(self, data, num_cluster):
        """
        use best of cluster
        :param df: time series dataset
        :param num_cluster:
        :return:cluster label
        """
        ks = KShape(n_clusters=num_cluster, n_init=5, verbose=True, random_state=self.seed)
        y_pred = ks.fit_predict(data)
        return y_pred

    def plot_best_shape(self, data, num_cluster):
        """
        time series cluster plot
        :param df:
        :param num_cluster:
        :return:
        """
        ks = KShape(n_clusters=num_cluster, n_init=5, verbose=True, random_state=self.seed)
        y_pred = ks.fit_predict(data)
        for yi in range(num_cluster):
            for xx in data[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.3)
            plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
            plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                     transform=plt.gca().transAxes)
            plt.tight_layout()
            plt.show()


def main(data):
    seed = 666
    data = pd.read_excel(data)
    print(data.head())
    pcts = Plot_Cluster_Time_Series(data, seed)
    input_df = pcts.read_data(data)

    k_shape, k_score = [], []
    for i in range(2, 7):
        shape_pred = pcts.cal_k_shape(input_df, i)
        score = pcts.shape_score(input_df, shape_pred)
        k_score.append(score)
        k_shape.append(i)

    dict_shape = dict(zip(k_shape, k_score))
    best_shape = sorted(dict_shape.items(), key=lambda x: x[1], reverse=True)[0][0]
    best_shape_list = sorted(dict_shape.items(), key=lambda x: x[1], reverse=True)[:][0]
    print('best_shape :', best_shape)
    print('dict_shape:', dict_shape)
    fin_label = pcts.cal_k_shape(input_df, best_shape)

    # fin_cluster = pd.DataFrame({"id": multi_id, "cluster_label": fin_label})
    pcts.plot_best_shape(input_df, best_shape)
    pcts.plot_best_shape(input_df, 5)
    pcts.plot_elbow(input_df)
    return fin_label


if __name__ == '__main__':
    fin_cluster = main()

# In[ ]:




