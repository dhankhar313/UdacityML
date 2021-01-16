"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
import pickle

sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn import tree
from sklearn.metrics import accuracy_score

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
# your code goes here #

try:
    with open('model.pickle', 'rb') as file:
        clf = pickle.load(file)
        print('Opening Saved Model, so no training time!!')
        print('Saved Model Training Time: 46 s')
except:
    clf = tree.DecisionTreeClassifier(min_samples_split=40)
    t0 = time()
    clf.fit(features_train, labels_train)
    print("Training time:", round(time() - t0, 3), "s")
    with open('model.pickle', 'wb') as file:
        pickle.dump(clf, file)

prediction = clf.predict(features_test)
accuracy = accuracy_score(labels_test, prediction)
print("Model Accuracy:", accuracy)
print(len(features_train[0]))

#########################################################
