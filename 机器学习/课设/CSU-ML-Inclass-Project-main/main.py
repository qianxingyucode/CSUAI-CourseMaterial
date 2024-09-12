import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from sklearn.datasets import load_breast_cancer  # 乳腺癌数据集 469，30
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.metrics import accuracy_score  # 准确率
from sklearn.metrics import f1_score        # F1值
from sklearn.metrics import roc_curve, auc  # ROC曲线和AUC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay    # 混淆矩阵


# 决策树桩
class DecisionStump:
    def __init__(self):
        # 基于划分阈值决定
        self.label = 1
        # 特征索引
        self.feature_index = None
        # 特征划分阈值
        self.threshold = None
        # 权重
        self.alpha = None


class AdaBoost:
    # 弱分类器集合
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
        self.estimators = []

    # AdaBoost算法
    def fit(self, X, y):
        # 获取样本数、维数
        m, n = X.shape
        # 初始化权重相等
        w = np.full(m, 1 / m)
        for _ in range(self.n_estimators):
            # 训练一个弱分类器：决策树桩
            estimator = DecisionStump()
            # 设定一个最小化误差率（初始设为∞）
            min_error = float('inf')
            # 遍历数据集特征，根据最小分类误差率选择最优特征作为分类依据
            for i in range(n):
                # 获取特征值
                values = np.expand_dims(X[:, i], axis=1)
                # 特征取值去重
                unique_values = np.unique(values)
                # 尝试将每一个特征值作为分类标准
                for threshold in unique_values:
                    label = 1
                    # 初始化所有预测值为1
                    pred = np.ones(np.shape(y))
                    # 小于分类阈值的预测值为-1
                    pred[X[:, i] < threshold] = -1
                    # 计算误差率
                    error = sum(w[y != pred])
                    # 误差率大于0.5，直接翻转正负预测
                    if error > 0.5:
                        error = 1 - error
                        label = -1
                    # 保存最小误差率相关参数
                    if error < min_error:
                        estimator.label = label
                        estimator.threshold = threshold
                        estimator.feature_index = i
                        min_error = error
            # 计算基分类器的权重
            estimator.alpha = 0.5 * np.log((1.0 - min_error) / max(min_error, 1e-12))
            # 初始化所有的预测值为 1
            preds = np.ones(np.shape(y))
            # 获取所有小于阈值的负类索引
            negative_idx = (estimator.label * X[:, estimator.feature_index] < estimator.label * estimator.threshold)
            # 将负类设置为 -1
            preds[negative_idx] = -1
            # 更新样本权重
            w *= np.exp(-estimator.alpha * y * preds)
            w /= np.sum(w)
            # 保存该弱分类器
            self.estimators.append(estimator)

    # 随机森林算法
    def fit_RF(self, X, y):
        # 获取样本数、维数
        m, n = X.shape
        # 初始化权重相等
        w = np.full(m, 1 / m)
        for _ in range(self.n_estimators):
            # 训练一个弱分类器：决策树桩
            estimator = DecisionStump()
            # 设定一个最小化误差率（初始设为∞）
            min_error = float('inf')
            k = 5
            rand = np.random.randint(0, n, k)
            # 遍历数据集特征，根据最小分类误差率选择最优特征作为分类依据
            for i in rand:
                # 获取特征值
                values = np.expand_dims(X[:, i], axis=1)
                # 特征取值去重
                unique_values = np.unique(values)
                # 尝试将每一个特征值作为分类阈值
                for threshold in unique_values:
                    label = 1
                    # 初始化所有预测值为1
                    pred = np.ones(np.shape(y))
                    # 小于分类阈值的预测值为-1
                    pred[X[:, i] < threshold] = -1
                    # 计算误差率
                    error = sum(w[y != pred])
                    # 误差率大于0.5，直接翻转正负预测
                    if error > 0.5:
                        error = 1 - error
                        label = -1
                    # 保存最小误差率相关参数
                    if error < min_error:
                        estimator.label = label
                        estimator.threshold = threshold
                        estimator.feature_index = i
                        min_error = error
            # 计算基分类器的权重
            estimator.alpha = 0.5 * np.log((1.0 - min_error) / max(min_error, 1e-12))
            # 初始化所有的预测值为 1
            preds = np.ones(np.shape(y))
            # 获取所有小于阈值的负类索引
            negative_idx = (estimator.label * X[:, estimator.feature_index] < estimator.label * estimator.threshold)
            # 将负类设置为 -1
            preds[negative_idx] = -1
            # 更新样本权重
            w *= np.exp(-estimator.alpha * y * preds)
            w /= np.sum(w)
            # 保存该弱分类器
            self.estimators.append(estimator)

    def predict(self, X):
        m = len(X)
        y_pred = np.zeros((m, 1))
        # 计算每个弱分类器的预测值
        for estimator in self.estimators:
            # 初始化所有预测值为1
            predictions = np.ones(np.shape(y_pred))
            # 获取所有小于阈值的负类索引
            negative_idx = (estimator.label * X[:, estimator.feature_index] < estimator.label * estimator.threshold)
            # 将负类设为 -1
            predictions[negative_idx] = -1
            # 对每个弱分类器的预测结果乘以alpha加权后进行累加
            y_pred += estimator.alpha * predictions
        y_score = y_pred
        # 返回最终预测结果
        y_pred = np.sign(y_pred).flatten()
        return y_pred, y_score


def calculate_recall_fpr_tpr(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    Tp = 0
    Fp = 0
    Tn = 0
    Fn = 0
    for label, pred in zip(y_true, y_pred):
        if (label == -1) and (pred == -1):
            Tp = Tp + 1
        elif (label == 1) and (pred == -1):
            Fp = Fp + 1
        elif (label == 1) and (pred == 1):
            Tn = Tn + 1
        elif (label == -1) and (pred == 1):
            Fn = Fn + 1
        else:
            print('Labels error!')
            return -1
    recall = Tp / (Tp + Fn)
    fpr = Fp / (Fp + Tn)
    tpr = Tp / (Tp + Fn)
    print("Recall:", recall)
    print("FPR:", fpr)
    print("TPR:", tpr)


def ROC(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def conf_matrix(y_test, y_pred, t):
    cm = confusion_matrix(y_test, y_pred)  # 混淆矩阵
    print("Confusion matrix of Label is \n", cm)
    # confusion_matrix(混淆矩阵), display_labels(标签名称列表)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
    # 画出混淆矩阵
    disp.plot()
    plt.title(f"Confusion matrix_{t}")
    # 保存
    # plt.savefig(f"Confusion_matrix_RF{t}")
    # 显示
    plt.show()


if __name__ == '__main__':
    # 获取乳腺癌数据集
    X, y = load_breast_cancer(return_X_y=True)
    # 数据集保存
    # cancerdata = load_breast_cancer()
    # cancerdata = pd.DataFrame(cancerdata['data'], columns=cancerdata['feature_names'])
    # cancerdata.to_csv('cancer.csv')

    # 将标签转换为 1 和 -1
    y_ = y.copy()
    y_[y == 0] = -1
    y_ = y_.astype(float)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y_, test_size=0.3, random_state=43)
    # 进行不同数量弱分类器的集成学习
    for t in [1, 5, 15]:
        print("———————————————AdaBoost———————————————————")
        print("弱分类器数量：", t)
        # 创建 Adaboost 模型实例
        clf = AdaBoost(n_estimators=t)
        # 模型拟合
        clf.fit(X_train, y_train)
        # 模型预测
        y_pred, y_score = clf.predict(X_test)
        # 计算模型预测的分类准确率
        accuracy = accuracy_score(y_test, y_pred)
        # 计算模型预测的recall, FPR, TPR
        calculate_recall_fpr_tpr(y_test, y_pred)
        # 计算模型预测的F1值
        F1 = f1_score(y_test, y_pred, labels=None, pos_label=1,
                      average='binary', sample_weight=None, zero_division="warn")
        print("准确率:", accuracy)
        print("F1:", F1)
        ROC(y_test, y_score)
        conf_matrix(y_test, y_pred, t)

    print("*************************************\n"
          "*************************************\n")
    # 进行不同数量弱分类器的集成学习
    for t in [1, 5, 15]:
        print("———————————————R F———————————————————")
        print("弱分类器数量：", t)
        # 创建模型实例
        clf = AdaBoost(n_estimators=t)
        # 模型拟合
        clf.fit_RF(X_train, y_train)
        # 模型预测
        y_pred, y_score = clf.predict(X_test)
        # 计算模型预测的分类准确率
        accuracy = accuracy_score(y_test, y_pred)
        # 计算模型预测的recall, FPR, TPR
        calculate_recall_fpr_tpr(y_test, y_pred)
        # 计算模型预测的F1值
        F1 = f1_score(y_test, y_pred, labels=None, pos_label=1,
                      average='binary', sample_weight=None, zero_division="warn")
        print("准确率:", accuracy)
        print("F1:", F1)
        ROC(y_test, y_score)
        conf_matrix(y_test, y_pred, t)
