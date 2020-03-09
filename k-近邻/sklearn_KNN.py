import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 数据
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

# 实例化模型，设置邻居数k
clf = KNeighborsClassifier(n_neighbors=3)
# 输入数据
clf.fit(X_train, y_train)
# 预测
ret = clf.predict(X_test)

print("预测结果：", ret)
