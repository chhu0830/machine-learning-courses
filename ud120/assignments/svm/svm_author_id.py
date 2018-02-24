#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC

clf = SVC(kernel="linear")          # 88.5% with 1% data, 98.4% with 100% data
clf = SVC(kernel="rbf")             # 61.6% with 1% data
clf = SVC(kernel="rbf", C=10000)    # 89.2% with 1% data, 99.1% with 100% data

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

print "accuracy:", round(clf.score(features_test, labels_test), 3)*100, "%"

#########################################################


