import sys
import collections
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial.distance import squareform
import xlrd
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import time
plt.style.use('bmh')

# 外部调用
# m = KnnDtw(n_neighbors=5, max_warping_window=10)
# m.kd_predict('D:/Essay/dtw_knn/x_test.xlsx',10)
# 内部调用
# train_model()

try:
    from IPython.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False


class KnnDtw(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays

    Arguments
    ---------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for KNN

    max_warping_window : int, optional (default = infinity)
        Maximum warping window allowed by the DTW dynamic
        programming function

    subsample_step : int, optional (default = 1)
        Step size for the timeseries array. By setting subsample_step = 2,
        the timeseries length will be reduced by 50% because every second
        item is skipped. Implemented by x[:, ::subsample_step]
        跳过几个
    """

    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step

    def fit(self, x, l, window):
        self.x = x
        self.l = l
        self.window = window
        # x 训练数据，l 对应label
        """Fit the model using x as training data and l as class labels

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer

        l : array of shape [n_samples]
            Training labels for input into KNN classifier
        """

    def d(self, x, y):
        # x = list(map(float, x))
        # y = list(map(float, y))

        return np.sum((x - y) ** 2)
# 多维dtw
    def dtw_distance(self, ts_a, ts_b):
        M, N = np.shape(ts_a)[1], np.shape(ts_b)[1]
        cost = sys.maxsize * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = self.d(ts_a[:, 0], ts_b[:, 0])
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + self.d(ts_a[:, i], ts_b[:, 0])
        for j in range(1, N):
            cost[0, j] = cost[0, j - 1] + self.d(ts_a[:, 0], ts_b[:, j])
        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(max(1, i - self.max_warping_window), min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
                cost[i, j] = min(choices) + self.d(ts_a[:, i], ts_b[:, j])

        # Return DTW distance given window
        return cost[-1, -1]


    def _dtw_distance(self, ts_a, ts_b, d=lambda x, y: abs(x - y)):
        """Returns the DTW similarity distance between two 2-D
        timeseries numpy arrays.

        Arguments
        ---------
        ts_a, ts_b : array of shape [n_samples, n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared
        两个要比较DTW距离的数组

        d : DistanceMetric object (default = abs(x-y))
            the distance measure used for A_i - B_j in the
            DTW dynamic programming function

        Returns
        -------
        DTW distance between A and B
        """

        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = sys.maxsize * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(max(1, i - self.max_warping_window),
                            min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        # Return DTW distance given window
        # 返回最后一个值
        return cost[-1, -1]

    def _dist_matrix(self, x, y):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]

        y : array of shape [n_samples, n_timepoints]

        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """

        # Compute the distance matrix
        dm_count = 0

        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if (np.array_equal(x, y)):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)

            p = ProgressBar(dm.shape[0])

            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = self.dtw_distance(x[i, ::self.subsample_step],
                                                      y[j, ::self.subsample_step])

                    dm_count += 1
                    # p.animate(dm_count)

            # Convert to squareform
            # 如果输入的是简洁的距离矩阵，将返回冗余矩阵；
            # 如果输入的是冗余的距离矩阵，将返回简洁的距离矩阵
            dm = squareform(dm)
            return dm

        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0]))
            dm_size = x_s[0] * y_s[0]

            p = ProgressBar(dm_size)

            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    dm[i, j] = self.dtw_distance(x[i, :, :],
                                                 y[j, :, :])
                    # Update progress bar
                    dm_count += 1
                    # p.animate(dm_count)

            return dm

    def predict(self, x):
        """Predict the class labels or probability estimates for
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified

        Returns
        -------
          2 arrays representing:
              (1) the predicted class labels
              (2) the knn label count probability
        """

        dm = self._dist_matrix(x, self.x)

        # Identify the k nearest neighbors
        # argsort函数是对数组中的元素进行从小到大排序，并返回相应序列元素的数组下标
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.l[knn_idx]

        # Model Label
        # mode返回传入数组/矩阵中最常出现的成员以及出现的次数,axis按列出现
        # 这里就是出现最多次的类型
        # mode_data[0]出现最多的值，mode_data[1]出现次数
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1] / float(self.n_neighbors)
        return mode_label.ravel(), mode_proba.ravel()


    def kd_predict(self,path,window):
        xtr = xlrd.open_workbook('D:/Essay/dtw_knn/x_train.xlsx')
        ytr = xlrd.open_workbook('D:/Essay/dtw_knn/y_train.xlsx')
        xte = xlrd.open_workbook(path)
        sheet1 = xtr.sheet_by_index(0)  # 返回第一个表的对象
        sheet2 = ytr.sheet_by_index(0)  # 返回第一个表的对象
        sheet3 = xte.sheet_by_index(0)  # 返回第一个表的对象

        data1 = self.x_preprocess(sheet1, window)
        data2 = self.y_preprocess(sheet2, window)
        data3 = self.x_preprocess(sheet3, window)
        # Convert to numpy for efficiency
        x_train = np.array(data1)
        y_train = np.array(data2)
        x_test = np.array(data3)
        labels = {1: '正常', 2: '全部遮挡', 3: '部分遮挡',
                  4: '断路'}

        m = KnnDtw(n_neighbors=self.n_neighbors, max_warping_window=self.max_warping_window)
        m.fit(x_train, y_train,window)
        label, proba = m.predict(x_test)
        print('label', label)
        print('proba', proba)
        return label,proba
        # print (classification_report(label, y_test[::10],
        #                       target_names=[l for l in labels.values()]))
        #
        # conf_mat = confusion_matrix(label, y_test[::10])

        # fig = plt.figure(figsize=(6, 6))
        # width = np.shape(conf_mat)[1]
        # height = np.shape(conf_mat)[0]
        #
        # res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
        # for i, row in enumerate(conf_mat):
        #     for j, c in enumerate(row):
        #         if c > 0:
        #             plt.text(j - .2, i + .1, c, fontsize=16)
        # plt.rcParams['axes.grid'] = False
        # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        # cb = fig.colorbar(res)
        # plt.title('Confusion Matrix')
        # _ = plt.xticks(range(4), [l for l in labels.values()], rotation=90)
        # _ = plt.yticks(range(4), [l for l in labels.values()])
        # plt.show()

    def x_preprocess(self,sheet, window):
        '''
        :param sheet: 输入的sheet页
        :param window:  窗口大小，多少个时间点属于一个ts
        :return: 输出 （TS个数*维数*window）结构的时间序列数组
        '''
        x = []
        for r in range(1, sheet.nrows):
            # 添加一整行
            x.append(sheet.row_values(r,1,sheet.ncols-1))
        # x.shape()=(n_timespoints,n_dimension)
        # 转置
        x = list(map(list, zip(*x)))
        re = []
        start = 0
        end = window - 1
        while end < len(x[0]):
            sub = []
            for i in range(len(x)):
                sub.append(x[i][start:end])
            re.append(sub)
            start = start + window
            end = end + window
        return re

    def y_preprocess(self,sheet, window):
        '''
        :param sheet: 输入的sheet页
        :param window:  窗口大小，多少个时间点属于一个ts
        :return: 输出 （TS个数*维数*window）结构的时间序列数组
        '''
        x = []
        for r in range(1, sheet.nrows):
            x.append(sheet.cell(r, 1).value)
        # x.shape()=(n_timespoints,n_dimension)
        # print("test行数，列数",len(x),'\n',len(x[0]))
        # x = list(map(list, zip(*x)))
        re = []
        start = 0
        end = window - 1
        while end < len(x):
            mode_data = mode(x[start:end])
            # print(x[start:end])
            re.append(mode_data[0][0])
            start = start + window
            end = end + window
        return re

class ProgressBar:
    """This progress bar was taken from PYMC
    """

    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            pass
#             self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        print ('\r', self)
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
                        (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)

def x_preprocess(sheet,window):
    '''
    :param sheet: 输入的sheet页
    :param window:  窗口大小，多少个时间点属于一个ts
    :return: 输出 （TS个数*维数*window）结构的时间序列数组
    去掉第一列times
    '''
    x = []
    for r in range(1, sheet.nrows):
        # 添加一整行,第一列是时间不要
        x.append(sheet.row_values(r,1,sheet.ncols-1))
    # x.shape()=(n_timespoints,n_dimension)
    # 转置
    x = list(map(list, zip(*x)))
    re = []
    start = 0
    end = window - 1
    while end < len(x[0]):
        sub = []
        for i in range(len(x)):
            sub.append(x[i][start:end])
        re.append(sub)
        start = start + window
        end = end + window
    return re
def y_preprocess(sheet,window):
    '''
    :param sheet: 输入的sheet页
    :param window:  窗口大小，多少个时间点属于一个ts
    :return: 输出 （TS个数*维数*window）结构的时间序列数组
    '''
    x = []
    #第一列是时间训练时不要
    for r in range(1, sheet.nrows):
        x.append(sheet.cell(r, 1).value)
    # x.shape()=(n_timespoints,n_dimension)
    # print("test行数，列数",len(x),'\n',len(x[0]))
    # x = list(map(list, zip(*x)))
    re = []
    start = 0
    end = window - 1
    while end < len(x):
        mode_data = mode(x[start:end])
        # print(x[start:end])
        re.append(mode_data[0][0])
        start = start + window
        end = end + window
    return re

def train_model(window):
        xtr = xlrd.open_workbook('D:/Essay/dtw_knn/x_train.xlsx')
        xte = xlrd.open_workbook('D:/Essay/dtw_knn/x_test.xlsx')
        ytr = xlrd.open_workbook('D:/Essay/dtw_knn/y_train.xlsx')
        yte = xlrd.open_workbook('D:/Essay/dtw_knn/y_test.xlsx')
        sheet1 = xtr.sheet_by_index(0)  # 返回第一个表的对象
        sheet2 = xte.sheet_by_index(0)  # 返回第一个表的对象
        sheet3 = ytr.sheet_by_index(0)  # 返回第一个表的对象
        sheet4 = yte.sheet_by_index(0)  # 返回第一个表的对象

        data1 = x_preprocess(sheet1,window)
        data2 = x_preprocess(sheet2,window)
        data3 = y_preprocess(sheet3,window)
        data4 = y_preprocess(sheet4,window)
        # Convert to numpy for efficiency
        x_train = np.array(data1)
        x_test = np.array(data2)
        y_train = np.array(data3)
        y_test = np.array(data4)

        # # Create empty lists
        # x_train = []
        # y_train = []
        # x_test = []
        # y_test = []
        # # 创建一个for循环迭代读取xls文件每行数据的, 从第二行开始是要跳过标题行
        # for r in range(1, sheet1.nrows):
        #     x_train.append(sheet1.row_values(r, 0, sheet1.ncols))
        # for r in range(1, sheet2.nrows):
        #     x_test.append(sheet2.row_values(r, 0, sheet2.ncols))
        # for r in range(1, sheet3.nrows):
        #     y_train.append(sheet3.row_values(r, 0, sheet3.ncols))
        # for r in range(1, sheet4.nrows):
        #     y_test.append(sheet4.row_values(r, 0, sheet4.ncols))

        labels = {1: '正常', 2: '全部遮挡', 3: '部分遮挡',
                  4: '断路'}

        m = KnnDtw(n_neighbors=1, max_warping_window=10)
        # m.fit(x_train[::10], y_train[::10], 10)
        # label, proba = m.predict(x_test[::10])
        m.fit(x_train,y_train, window=10)
        label, proba = m.predict(x_test)

        print(classification_report(y_test, label,
                                    target_names=[l for l in labels.values()]))

        conf_mat = confusion_matrix(y_test,label)
        score = accuracy_score(y_test,label)
        fig = plt.figure(figsize=(6, 6))
        width = np.shape(conf_mat)[1]
        height = np.shape(conf_mat)[0]

        res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
        for i, row in enumerate(conf_mat):
            for j, c in enumerate(row):
                if c > 0:
                    plt.text(j - .2, i + .1, c, fontsize=16)
        plt.rcParams['axes.grid'] = False
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        cb = fig.colorbar(res)
        plt.title('Confusion Matrix')
        _ = plt.xticks(range(4), [l for l in labels.values()], rotation=90)
        _ = plt.yticks(range(4), [l for l in labels.values()])
        plt.show()

if __name__ == '__main__':
# m = KnnDtw(n_neighbors=5, max_warping_window=10)
    # m.kd_predict('D:/Essay/dtw_knn/x_test.xlsx',10)
    acc_s = []
    windows = [1, 2, 5, 10, 50]

    for w in windows:
        s = train_model(w)
        acc_s.append(s)

    fig = plt.figure(figsize=(12, 5))
    _ = plt.plot(windows, acc_s, lw=4)
    plt.title('准确率随窗口大小的变化')
    plt.ylabel('模型验证准确率')
    plt.xlabel('窗口大小')
    plt.xscale('log')
    plt.show()
# #测试dtw
# fig = plt.figure(figsize=(12,4))
# plt.plot(time, amplitude_a, label='A')
# _ = plt.plot(time, amplitude_b, label='B')
# _ = plt.title('DTW distance between A and B is %.2f' % distance)
# _ = plt.ylabel('Amplitude')
# _ = plt.xlabel('Time')
# _ = plt.legend()
# plt.show()

# Mapping table for classes
# Import the HAR dataset
# x_train_file

