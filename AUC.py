from sklearn.metrics import roc_auc_score
import numpy
import matplotlib.pyplot as plt
from scipy.integrate import simps

y_true = [0, 0, 1, 1, 0]

y_score = [0.1, 0.4, 0.35, 0.8, 0.6]


print("1. 自己实现的代码")
def auc(y_true, y_score):
    n = len(y_true)
    pos, neg = sum(y_true), n - sum(y_true)
    "zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。"
    # 对预测分数和真实标签进行排序，根据预测分数进行降序排列
    zip_true_score = list(zip(y_score, y_true))
    tuples = sorted(zip_true_score, reverse=True)
    tp, fp, fn, tn = 0, 0, 0, 0
    Pd, Fa = [], []
    # thresh = [1.8, 0.8, 0.4, 0.35, 0.1]
    thresh = [0.8, 0.6, 0.4, 0.35, 0.1]
    for thre in thresh:
        for i in range(n):
            if tuples[i][0] >= thre and tuples[i][1] == 1 :
                tp += 1
            if tuples[i][0] < thre and tuples[i][1] == 1 :
                fn += 1
            if tuples[i][0] >= thre and tuples[i][1] == 0 :
                fp += 1
            if tuples[i][0] < thre and tuples[i][1] == 0 :
                tn += 1

            # if tuples[i][0] >= thre and tuples[i][1] == 1 :
            #     tp += 1
            # if (1 - tuples[i][0]) >= thre and tuples[i][1] == 1 :
            #     fn += 1
            # if tuples[i][0] >= thre and tuples[i][1] == 0 :
            #     fp += 1
            # if (1 - tuples[i][0]) >= thre and tuples[i][1] == 0 :
            #     tn += 1
        if tp + fn == 0 or fp + tn == 0 :
            Pd.append(0)
            Fa.append(0)
        else:
            pd = tp / (tp + fn)
            fa = fp / (fp + tn)
            Pd.append(pd)
            Fa.append(fa)
    # plt.plot(Fa, Pd)
    # plt.title('ROC Curve 1')
    # plt.xlabel('Fa')
    # plt.ylabel('Pd')
    # plt.show()
    area = 0
    Pd = sorted(Pd)
    Fa = sorted(Fa)
    area_trapz = numpy.trapz(Pd, Fa)
    print("使用numpy.trapz()求解的曲线图的线下面积为：", area_trapz)
    for i in range(1, len(Fa)):
        # Calculate the area of the trapezoid formed by this point and the previous point.
        trap_area = (Fa[i] - Fa[i - 1]) * (Pd[i] + Pd[i - 1]) / 2.0
        area += trap_area
    return area
auc_our = auc(y_true, y_score)
print("使用梯度法则求解曲线图的线下面积为：", auc_our)
print("-----------------------------------------------------------")

print("2. Chatgpt实现的代码")
def roc_curve_ourshixian(y_true, y_score):
    fpr, tpr, thresholds = [], [], []
    total_positives = sum(y_true)
    total_negatives = len(y_true) - total_positives

    # sort predicted scores in descending order
    sorted_indices = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)
    sorted_y_true = [y_true[i] for i in sorted_indices]

    fp, tp = total_negatives, total_positives
    for i in range(len(sorted_y_true)):
        if sorted_y_true[i] == 1:
            tp -= 1
        else:
            fp -= 1

        fn = total_positives - tp
        tn = total_negatives - fp

        # append current fpr, tpr, and threshold values
        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))
        thresholds.append(y_score[sorted_indices[i]])
    area = 0
    fpr = sorted(fpr)
    tpr = sorted(tpr)
    area_trapz = numpy.trapz(tpr, fpr)
    print("使用numpy.trapz()求解的曲线图的线下面积为：", area_trapz)
    for i in range(1, len(fpr)):
        trap_area = (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
        area += trap_area
    return fpr, tpr, thresholds, area
fpr, tpr, thresholds, area = roc_curve_ourshixian(y_true, y_score)
print("使用梯度法则求解曲线图的线下面积为：", area)
# plt.plot(fpr, tpr)
# plt.title('ROC Curve 2')
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.show()

print("******************************************************")

print("3. 调用roc_curve()和auc()实现的代码")
from sklearn import metrics
# 根据真实标签和预测概率计算AUC
"y_score:该参数要求输入的是你正类的目标分数可以是正类的概率估计、置信值。注意！一定是正类"
def calculate_auc(y_true, y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
    # plt.plot(fpr, tpr)
    # plt.title('ROC Curve')
    # plt.xlabel('fpr')
    # plt.ylabel('tpr')
    # plt.show()
    auc = metrics.auc(fpr, tpr)
    return auc
auc = calculate_auc(y_true, y_score)
print(auc)
print("-----------------------------------------------------------")

print("4. 调用roc_auc_score()实现的代码")
"调用roc_auc_score()直接计算AUC"
from sklearn.metrics import roc_auc_score
auc2 = roc_auc_score(y_true, y_score)
print(auc2)

print("-----------------------------------------------------------")
# y_true = [0, 0, 1, 1, 0]
# y_score = [0.1, 0.4, 0.35, 0.8, 0.6]
print("5. chatgpt代码实现的代码")
"chatgpt代码"
def auc(y_true, y_score):
    # 计算样本总数
    n = len(y_true)
    # 计算正样本数和负样本数
    pos, neg = sum(y_true), n - sum(y_true)
    # 将预测得分和真实标签组成元组，并按预测得分从大到小排序
    tuples = sorted(zip(y_score, y_true), reverse=True)
    # 初始化真正率(tp_rate)和假正率(fp_rate)为0
    tp, fp = 0, 0
    tprate, fprate = [], []
    # 遍历每一个样本
    for i in range(n):
        # 如果该样本为正样本，将tp加1；否则将fp加1
        if tuples[i][1] == 1:
            tp += 1
        else:
            fp += 1
        # 计算当前真正率和假正率
        tprate.append(tp/pos)
        fprate.append(fp/neg)
    # 初始化AUC面积为0
    area = 0
    # 计算ROC曲线下的面积
    for i in range(1, n):
        area += (fprate[i]-fprate[i-1]) * (tprate[i]+tprate[i-1])/2
    # 返回AUC面积
    return area
auc1 = auc(y_true, y_score)
print(auc1)

def calculate_auc(y_true, y_score):
    """
    计算AUC
    :param y_true: 一个列表，表示样本的真实标签，用01表示
    :param y_score: 一个列表，表示模型预测为正样本的概率分数，取值在[0, 1]之内
    :return: AUC值
    """
    n = len(y_true)  # 样本总数
    pos = sum(y_true)  # 正样本数
    neg = n - pos  # 负样本数

    # 根据预测概率从大到小排序
    scores_labels = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)

    # 计算累积正样本数和累积负样本数
    cum_pos, cum_neg = 0, 0
    for i in range(n):
        if scores_labels[i][1] == 1:
            cum_pos += 1
        else:
            cum_neg += 1

    # 计算AUC
    auc_value = 0
    tp, fp = 0, 0
    for i in range(1, n):
        if scores_labels[i][0] != scores_labels[i-1][0]:
            auc_value += (cum_neg-fp) * (cum_pos-tp) / (pos * neg)
            tp_prev, fp_prev = tp, fp
        if scores_labels[i][1] == 1:
            tp += 1
        else:
            fp += 1
    auc_value += (cum_neg-fp+1) * (cum_pos-tp) / (pos * neg)
    return auc_value

auc = calculate_auc(y_true, y_score)
print("ttttt", auc)