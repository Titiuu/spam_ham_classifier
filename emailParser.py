# -*- coding: utf-8 -*-
import email
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('porter_test')

stopwords = set(nltk.corpus.stopwords.words('english'))
PS = PorterStemmer()


# 将邮件的不同部分合并成一个字符串列表
def flatten_to_string(parts):
    ret = []
    if type(parts) == str:
        ret.append(parts)
    elif type(parts) == list:
        for part in parts:
            ret += flatten_to_string(part)
    elif parts.get_content_type == 'text/plain':
        ret += parts.get_payload()
    return ret


# 邮件内容处理
def extract_email_text(path):
    # 读邮件
    with open(path, errors='ignore') as f:
        msg = email.message_from_file(f)
    if not msg:
        return ""

    # 主题
    subject = msg['Subject']
    if not subject:
        subject = ""

    # 正文，去停用词，提取词干
    body = ' '.join(m for m in flatten_to_string(msg.get_payload()) if type(m) == str)
    if not body:
        body = ""
    sent_tokens = sent_tokenize(body)
    filtered_body = []
    for sent in sent_tokens:
        word_tokens = word_tokenize(sent)
        filtered_sents = [PS.stem(word) for word in word_tokens if word not in stopwords]
        filtered_sents = ' '.join(filtered_sents)
        filtered_body.append(filtered_sents)
    body = ' '.join(filtered_body)
    return subject + ' ' + body

