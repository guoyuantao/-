import numpy as np
from sklearn.linear_model import Perceptron

# 训练数据
X_train = np.array([
    [3, 3],
    [4, 3],
    [1, 1],
])
y_train = np.array([1, 1, -1])

# 实例化模型
model = Perceptron()

# 训练模型
model.fit(X_train, y_train)

# 展示结果
print('参数：')
print("w:", model.coef_, '\n', "b:", model.intercept_, '\n', 'iter:', model.n_iter_)

# 准确率
ret = model.score(X_train, y_train)
print("准确率：", ret)