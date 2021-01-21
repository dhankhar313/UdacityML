import pickle
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

numpy.random.seed(42)

# The words (features) and authors (labels), already largely processed.
# These files should have been created from the previous (Lesson 10)
# mini-project.
word_data = pickle.load(open("../text_learning/your_word_data.pkl", "rb"))
authors = pickle.load(open("../text_learning/your_email_authors.pkl", "rb"))

# test_size is the percentage of events assigned to the test set (the
# remainder go into training)
# feature matrices changed to dense representations for compatibility with
# classifier functions in versions 0.15.2 and earlier

features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1,
                                                                            random_state=42)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train).toarray()
features_test = vectorizer.transform(features_test).toarray()

# a classic way to overfit is to use a small number
# of data points and a large number of features;
# train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train = labels_train[:150]
# your code goes here
clf = DecisionTreeClassifier(min_samples_split=2)
clf.fit(features_train, labels_train)
print("Accuracy: ", clf.score(features_test, labels_test))
print("Accuracy: ", accuracy_score(labels_test, clf.predict(features_test)))
temp = list(clf.feature_importances_)
features = [i for i in temp if i >= 0.2]
print("Importance above 0.2:", features)
print("Index of Important feature: ", temp.index(features[0]))
print(vectorizer.get_feature_names()[temp.index(features[0])])
