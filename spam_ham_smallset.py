import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import re
classifier_mb = MultinomialNB()
classifier_bb = BernoulliNB()


# Loading the data set
email_data = pd.read_csv("./smallset/email/email/data.csv", usecols=["label", "content"], encoding="ISO-8859-1")
# print(email_data)
# print('######################################')
# print(email_data['content'][0])


# 去除符号和停用词的函数，停用词默认两字单词如‘it to’等，后续效果不好可以再考虑使用NLTK处理
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+", " ", i).lower()
    i = re.sub("[0-9" "]+", " ", i)
    w = []
    for word in i.split(" "):
        if len(word) > 2:
            w.append(word)
    return " ".join(w)


# 分成词向量的函数
def split_into_words(i):
    return i.split(" ")


# 除去符号和停用词
email_data['content'] = email_data['content'].apply(cleaning_text)

# 分割数据集为训练集和测试集
email_train, email_test = train_test_split(email_data, test_size=0.2)

# 分词并计数，向量化
emails_bow = CountVectorizer(analyzer=split_into_words).fit(email_data['content'])

# 学习完将数据集、训练集、测试集转换为矩阵
all_emails_matrix = emails_bow.transform(email_data['content'])
train_emails_matrix = emails_bow.transform(email_train['content'])
test_emails_matrix = emails_bow.transform(email_test['content'])

# Tfidf学习并转换为矩阵
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)
train_tfidf = tfidf_transformer.transform(train_emails_matrix)
test_tfidf = tfidf_transformer.transform(test_emails_matrix)

# Multinomial Naive Bayes
classifier_mb.fit(train_tfidf, email_train['label'])
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == email_test['label'])

# BernoulliNB Naive Bayes
classifier_bb.fit(train_tfidf, email_train['label'])
test_pred_g = classifier_bb.predict(test_tfidf)
accuracy_test_g = np.mean(test_pred_g == email_test['label'])

print('多项式模型准确率为 {:.1%}'.format(accuracy_test_m))
print('伯努利模型准确率为 {:.1%}'.format(accuracy_test_g))



