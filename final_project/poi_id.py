import sys
import pickle
from time import time

sys.path.append("../tools/")
sys.path.append("../outliers/")

from feature_format import featureFormat, targetFeatureSplit
from outlier_cleaner import outlierCleaner
from tester import dump_classifier_and_data
from matplotlib.pyplot import scatter
from sklearn.decomposition import PCA

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi', 'salary', 'from_poi_to_this_person', 'from_this_person_to_poi', 'bonus',
                 'total_stock_value']  # You will need to use more features

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# print(data_dict[list(data.keys())[0]])
# Extract features and labels from dataset for local testing
data = list(featureFormat(data_dict, features_list, sort_keys=True))
print('Total data points: ', len(data_dict))
print("After processing features....")
print("Data points with Outliers: ", len(data))
# Task 2: Remove outliers
factor = [0, 5, 4, 8, 4, 5]
for i in range(1, 6):
    for j in range(factor[i]):
        temp = [x[i] for x in data]
        idx = temp.index(max(temp))
        del data[idx]
print("Data points without Outliers: ", len(data), '\n')
labels, features = targetFeatureSplit(data)

# Task 3: Create new feature(s)
# pca = PCA(n_components=2, whiten=True).fit(features)
# features_pca = pca.transform(features)
# Store to my_dataset for easy export below.
# Task 4: Try a variety of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

clf1 = GaussianNB()
# svm_param = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
# clf2 = GridSearchCV(SVC(kernel='linear'), svm_param)
dt_params = {'min_samples_split': [2, 10, 30, 50, 100]}
clf = GridSearchCV(DecisionTreeClassifier(), dt_params)
knn_params = {'n_neighbors': [10, 30, 50]}
clf2 = GridSearchCV(KNeighborsClassifier(), knn_params)

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

classifiers = [clf1, clf, clf2]
for i, j in enumerate(classifiers):
    print(f'Classifier {i + 1} Training...')
    t0 = time()
    j.fit(features_train, labels_train)
    print(f'Trained in {round(time() - t0, 3)}s')
    try:
        print(j.best_estimator_, '\n')
    except:
        print('\n')

print('Naive Bayes Accuracy: ', clf1.score(features_test, labels_test))
# print('SVM Accuracy: ', clf2.score(features_test, labels_test))
print('Decision Tree Accuracy: ', clf.score(features_test, labels_test))
print('KNN Accuracy: ', clf2.score(features_test, labels_test))
# pickle.dump(clf2, open('svm_model.pickle', 'wb'))
# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, data_dict, features_list)
