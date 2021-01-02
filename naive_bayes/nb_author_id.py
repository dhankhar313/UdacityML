""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
import sys
from time import time
import pickle

sys.path.append(r"C:\Users\RahulDhankhar\PycharmProjects\UdacityML\tools")
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from email_preprocess import preprocess

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
# your code goes here ###

clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print("Training time:", round(time() - t0, 3), "s")
prediction = clf.predict(features_test)
accuracy = accuracy_score(labels_test, prediction)
print("Model Accuracy:", accuracy)
with open('model.pickle', 'wb') as file:
    pickle.dump(clf, file)
#########################################################
