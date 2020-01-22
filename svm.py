#coding:utf-8

import numpy as np
from sklearn import datasets

#手書き数字データの読み込み
digits = datasets.load_digits()

#3と8のデータ位置を求める
flag_1_8 = (digits.target == 1) + (digits.target == 8)

#3と8のデータを取得
images = digits.images[flag_1_8]
labels = digits.target[flag_1_8]

#3と8の画像データを1次元化
images = images.reshape(images.shape[0],-1)

from sklearn import svm

images = images.reshape(images.shape[0],-1)

#分類器の生成
n_samples = len(flag_1_8[flag_1_8])
train_size = int(n_samples * 3 / 5)
classifier = svm.SVC(C=1.0, gamma=0.001, kernel='rbf')
classifier.fit(images[:train_size], labels[:train_size]) 

#性能評価
from sklearn import metrics

expected = labels[train_size:]
predicted = classifier.predict(images[train_size:])

print '正答率\n',metrics.accuracy_score(expected,predicted)
print '\n混同行列\n',metrics.confusion_matrix(expected,predicted)
print '\n適合率\n',metrics.precision_score(expected,predicted,pos_label=1)
print '\n再現率\n',metrics.recall_score(expected,predicted,pos_label=1)
print '\nF値\n',metrics.f1_score(expected,predicted,pos_label=1)



