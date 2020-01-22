import numpy as np
from sklearn import datasets

digits = datasets.load_digits()

flag_3_8 = (digits.target == 3) + (digits.target == 8)

images = digits.images[flag_3_8]
labels = digits.target[flag_3_8]

images = images.reshape(images.shape[0],-1)

from sklearn import tree

images = images.reshape(images.shape[0],-1)

n_samples = len(flag_3_8[flag_3_8])
train_size = int(n_samples * 3 / 5)
classifier = tree.DecisionTreeClassifier()
classifier.fit(images[:train_size], labels[:train_size]) 


from sklearn import metrics

expected = labels[train_size:]
predicted = classifier.predict(images[train_size:])

print('Accuracy:\n',metrics.accuracy_score(expected,predicted))
