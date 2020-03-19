import numpy as np
import pandas as pd
import pydotplus

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.display import Image

def show(clf, features, y_types):
    """ 决策树的可视化 """
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=features,
                                    class_names=y_types,
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(r'DT_Show.png')

# 数据
features = ["age", "work", "house", "credit"]
X_train = pd.DataFrame([
                  ["青年", "否", "否", "一般"],
                  ["青年", "否", "否", "好"],
                  ["青年", "是", "否", "好"],
                  ["青年", "是", "是", "一般"],
                  ["青年", "否", "否", "一般"],
                  ["中年", "否", "否", "一般"],
                  ["中年", "否", "否", "好"],
                  ["中年", "是", "是", "好"],
                  ["中年", "否", "是", "非常好"],
                  ["中年", "否", "是", "非常好"],
                  ["老年", "否", "是", "非常好"],
                  ["老年", "否", "是", "好"],
                  ["老年", "是", "否", "好"],
                  ["老年", "是", "否", "非常好"],
                  ["老年", "否", "否", "一般"]
                  ])
y_train = pd.DataFrame(["否", "否", "是", "是", "否", "否", "否", "是", "是", "是", "是", "是", "是", "是", "否"])
# 因为数据是类别数据，需要进行处理，转化为数值型
# 处理X数据
Pre_X = preprocessing.LabelEncoder()
Pre_X.fit(np.unique(X_train))
X_train = X_train.apply(Pre_X.transform)
print(X_train)
# 处理y数据
Pre_y = preprocessing.LabelEncoder()
Pre_y.fit(np.unique(y_train))
y_train = y_train.apply(Pre_y.transform)
print(y_train)
# 实例化模型
clf = DecisionTreeClassifier()
# 训练
clf.fit(X_train, y_train)
# 可视化
show(clf, features, [str(k) for k in np.unique(y_train)])
# 测试数据
X_test = pd.DataFrame([["青年", "否", "是", "一般"]])
X_test = X_test.apply(Pre_X.transform)
# 预测
y_predict = clf.predict(X_test)
# 展示结果
X_show = [{features[i]: X_test.values[0][i]} for i in range(len(features))]
print("{0} 被分类为： {1}".format(X_show, Pre_y.inverse_transform(y_predict)))