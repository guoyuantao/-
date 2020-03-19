import numpy as np
import pandas as pd

# 贝叶斯类
class NaiveBayes():
    def __init__(self, lambda_):
        """ 初始化 """
        self.lambda_ = lambda_ # 参数，防止概率为0
        self.y_types_count = None  # 标签的数量
        self.y_proba = None  # 标签的概率
        self.x_types_proba = dict()   # 初始化一个字典，存储（xi 的编号,xi的取值，y的类型）：概率

    def fit(self, X_train, y_train):
        self.y_types = np.unique(y_train) # 求出y的所有类别
        # 将数据转为DataFrame格式
        X = pd.DataFrame(X_train)
        y = pd.DataFrame(y_train)
        # 统计y类型的数量 1    9,-1    6
        self.y_types_count = y[0].value_counts()
        # 计算类别概率, 1    0.597403,-1    0.402597
        self.y_types_proba = (self.y_types_count + self.lambda_) / (y.shape[0] + len(self.y_types) * self.lambda_)
        # （xi 的编号,xi的取值，y的类型）：概率的计算
        for idx in X.columns: # 遍历特征
            for j in self.y_types: # 选取每个y的类型
                # 选择所有y == j为真的数据点的第idx个特征的值，并对这些值进行（类型：数量）统计
                p_x_y = X[(y == j).values][idx].value_counts()
                # 计算概率
                for i in p_x_y.index:
                    self.x_types_proba[(idx, i, j)] = (p_x_y[i] + self.lambda_) / (self.y_types_count[j] + p_x_y.shape[0] * self.lambda_)

        print(self.x_types_proba)

    def predict(self, X_new):
        res = []
        for y in self.y_types:
            # y的先验概率
            p_y = self.y_types_proba[y]
            p_xy = 1
            for idx, x in enumerate(X_new):
                # 计算P(X=(x1,x2...xd)/Y=ck)
                p_xy *= self.x_types_proba[(idx, x, y)]
            res.append(p_y * p_xy)

        for i in range(len(self.y_types)):
            print("[{}]对应概率： {:.2%}".format(self.y_types[i], res[i]))

        # 返回最大后验概率对应的y值
        return self.y_types[np.argmax(res)]



def main():
    # 准备数据
    X_train = np.array([
        [1, "S"],
        [1, "M"],
        [1, "M"],
        [1, "S"],
        [1, "S"],
        [2, "S"],
        [2, "M"],
        [2, "M"],
        [2, "L"],
        [2, "L"],
        [3, "L"],
        [3, "M"],
        [3, "M"],
        [3, "L"],
        [3, "L"]
    ])
    y_train = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    # 实例化模型
    clf = NaiveBayes(lambda_=0.2)
    # 喂入数据，训练模型
    clf.fit(X_train, y_train)
    # 测试数据
    X_new = np.array([2, "S"])
    # 预测
    y_predict = clf.predict(X_new)
    # 展示结果
    print("{}被分类为：{}".format(X_new, y_predict))


if __name__ == '__main__':
    main()