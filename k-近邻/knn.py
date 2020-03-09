import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

class KNN:
    def __init__(self, X_train, y_train, k):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        dis_list = []
        for i in range(self.X_train.shape[0]):
            # 计算欧式距离
            distance = np.sqrt(((X_test - self.X_train[i]) ** 2).sum(axis=1))[0]
            # 与标签一起存入到列表中
            dis_list.append((distance, self.y_train[i]))
            # 排序
            dis_list.sort()
        # 取出y标签
        y_list = [dis_list[i][-1] for i in range(self.k)]
        # 计算每个标签的个数
        y_count = Counter(y_list).most_common()

        return y_count[0][0]


def draw(X_train, X_test):
    plt.figure()
    plt.scatter(X_train[:3, 0], X_train[:3, 1], c='g', marker='.')
    plt.scatter(X_train[3:6, 0], X_train[3:6, 1], c='r', marker='*')
    plt.scatter(X_test[:, 0], X_test[:, 1])
    plt.title('数据散点图')
    plt.xlabel('x坐标')
    plt.ylabel('y坐标')
    plt.show()

def main():
    # 训练数据
    X_train = np.array([[5, 4],
                       [9, 6],
                       [4, 7],
                       [2, 3],
                       [8, 1],
                       [7, 2]])
    y_train = np.array([1, 1, 1, -1, -1, -1])
    # 测试数据
    X_test = np.array([[5, 3]])
    # 画图，查看数据的情况
    draw(X_train, X_test)
    # 构建KNN模型
    # 设置不同的k值，测试对模型的影响
    for k in range(1, 6, 2):
        # 实例化模型
        clf = KNN(X_train, y_train, k)
        # 对测试数据进行预测
        ret = clf.predict(X_test)
        # 展示结果
        print("k = %d, result = %d" % (k, ret))


if __name__ == '__main__':
    main()