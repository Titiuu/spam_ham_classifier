# -*- coding: utf-8 -*-

import emailParser
import os
from time import time
# import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


os.getcwd()
DATA_DIR = "./bigdataset/trec07p/data"  # 数据集路径
LABELS_FILE = "./bigdataset/trec07p/full/index"  # 标签路径
EMAIL_NUM = 5000  # 邮件总数设置

labels = {}
# 读取标签
with open(LABELS_FILE) as f:
    num = 0
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0
        num += 1
        if num > EMAIL_NUM:
            break


# 读取邮件内容，返回邮件内容列表和标签列表
def read_email_files():
    x1 = []
    y1 = []
    for i in range(len(labels)):
        filename = 'inmail.' + str(i+1)
        email_str = emailParser.extract_email_text(os.path.join(DATA_DIR, filename))
        x1.append(email_str)
        y1.append(labels[filename])
    return x1, y1


X, y = read_email_files()

# print(pd.DataFrame(X).head())
# print(pd.DataFrame(y).head())

# 划分训测集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

# Tfidf分词提取特征，向量化
vectorizer = TfidfVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# 朴素贝叶斯分类器(多项式模型)
nb_start = time()
nb_clf = MultinomialNB()
nb_clf = nb_clf.fit(X_train_vector, y_train)
y_nbpred = nb_clf.predict(X_test_vector)
nb_end = time()
print('朴素贝叶斯准确率: {:.1%}'.format(accuracy_score(y_test, y_nbpred)))
print("精度:", precision_score(y_test, y_nbpred))
print("召回率:", recall_score(y_test, y_nbpred))
print("F1值:", f1_score(y_test, y_nbpred))
svcnf_matrix = confusion_matrix(y_test, y_nbpred)
print('混淆矩阵如下:')
print(svcnf_matrix)
print('运行时间为: {:.2f}'.format(nb_end - nb_start), '秒')

# 支持向量机分类器
sv_start = time()
sv_clf = svm.SVC(kernel='linear')
sv_clf = sv_clf.fit(X_train_vector, y_train)
y_svpred = sv_clf.predict(X_test_vector)
sv_end = time()
print('支持向量机准确率: {:.1%}'.format(accuracy_score(y_test, y_svpred)))
print("精度:", precision_score(y_test, y_svpred))
print("召回率:", recall_score(y_test, y_svpred))
print("F1值:", f1_score(y_test, y_svpred))
svcnf_matrix = confusion_matrix(y_test, y_svpred)
print('混淆矩阵如下:')
print(svcnf_matrix)
print('运行时间为: {:.2f}'.format(sv_end - sv_start), '秒')

# 随机森林分类器
forest_start = time()
forest_clf = RandomForestClassifier()
forest_clf = forest_clf.fit(X_train_vector, y_train)
y_forestpred = forest_clf.predict(X_test_vector)
forest_end = time()
print('随机森林准确率: {:.1%}'.format(accuracy_score(y_test, y_forestpred)))
print("精度:", precision_score(y_test, y_forestpred))
print("召回率:", recall_score(y_test, y_forestpred))
print("F1值:", f1_score(y_test, y_forestpred))
forestcnf_matrix = confusion_matrix(y_test, y_forestpred)
print('混淆矩阵如下:')
print(forestcnf_matrix)
print('运行时间为: {:.2f}'.format(forest_end - forest_start), '秒')




