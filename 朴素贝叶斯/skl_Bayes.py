import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import preprocessing

# 训练数据
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
y_train = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
# 数据预处理
enc = preprocessing.OneHotEncoder(categories='auto')
enc.fit(X_train)
X_train = enc.transform(X_train).toarray()
print(X_train)

# 实例化分类器
clf = MultinomialNB(alpha=0.0000001)
# 训练
clf.fit(X_train, y_train)
# 测试数据,并特征处理
X_new = np.array([[2, "S"]])
X_new = enc.transform(X_new).toarray()
# 预测
y_predict = clf.predict(X_new)
# 展示结果
print("{}被分类为：{}".format(X_new, y_predict))
print("预测的概率为：{}".format(clf.predict_proba(X_new)))
