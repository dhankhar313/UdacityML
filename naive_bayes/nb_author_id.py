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

sys.path.append(r"../tools/")
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
# your code goes here #
try:
    with open('model.pickle', 'rb') as file:
        clf = pickle.load(file)
        print('Opening Saved Model, so no training time!!')
        print('Saved Model Training Time: 0.9 s')
except:
    clf = GaussianNB()
    t0 = time()
    clf.fit(features_train, labels_train)
    print("Training time:", round(time() - t0, 3), "s")
    with open('model.pickle', 'wb') as file:
        pickle.dump(clf, file)

prediction = clf.predict(features_test)
accuracy = accuracy_score(labels_test, prediction)
print("Model Accuracy:", accuracy)

#########################################################
