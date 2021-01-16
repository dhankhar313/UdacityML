import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()

# the training data (features_train, labels_train) have both "fast" and "slow"
# points mixed together--separate them so we can give them different colors
# in the scatter plot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

# initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()

################################################################################
# your code here!  name your classifier object clf if you want the visualization code (prettyPicture) to show you the decision boundary
# KNN Accuracy = 93.6
# Random Forest Accuracy = 92
# Adaboost Accuracy = 92.4

# clf = KNeighborsClassifier(n_neighbors=20)
# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf = AdaBoostClassifier(n_estimators=100, random_state=0)

clf = KNeighborsClassifier(n_neighbors=20)
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
accuracy = accuracy_score(labels_test, prediction)
print("Accuracy: ", accuracy)

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
