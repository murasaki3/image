#coding:utf-8
from sklearn import datasets, cross_validation, svm, metrics
import numpy as np
from PIL import Image
image = Image.open("test4.png")
#image = image.resize((8, 8), Image.ANTIALIAS)
image.show()
