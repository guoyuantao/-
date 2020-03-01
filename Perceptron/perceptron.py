import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self):
        # 初始化参数
        self.w = None
        self.b = 0
        self.learning = 1

    def fit(self, X_train, y_train):
        self.w = np.zeros(X_train.shape[1])

        i = 0
        while i < X_train.shape[0]:  # 循环遍历每一个数据
            # 取出数据
            X = X_train[i]
            y = y_train[i]
            # 判断是否为误分类点
            if y * (np.dot(self.w, X) + self.b) <= 0:
                self.w = self.w + self.learning * (np.dot(y, X))
                self.b = self.b + self.learning * y
                # 误分类点更新之后，要从第一个数据开始遍历
                i = 0
            else:
                i += 1

    def draw(self, X, w, b):
        # 生产分离超平面上的两点
        X_new = np.array([[0], [6]])
        y_predict = -b - (w[0] * X_new) / w[1]
        # 绘制训练数据集的散点图
        plt.plot(X[:2, 0], X[:2, 1], "g*", label="1")
        plt.plot(X[2:, 0], X[2:, 0], "rx", label="-1")
        # 绘制分离超平面
        plt.plot(X_new, y_predict, "b-")
        # 设置两坐标轴起止值
        plt.axis([0, 6, 0, 6])
        # 设置坐标轴标签
        plt.xlabel('x1')
        plt.ylabel('x2')
        # 显示图例
        plt.legend()
        # 显示图像
        plt.show()

def main():
    # 训练数据
    X_train = np.array([
        [3, 3],
        [4, 3],
        [1, 1],
    ])
    y_train = np.array([1, 1, -1])

    # 感知机模型
    model = Perceptron()
    # 训练
    model.fit(X_train, y_train)
    # 画图
    model.draw(X_train, model.w, model.b)
    print(model.w, model.b)


if __name__ == '__main__':
    main()