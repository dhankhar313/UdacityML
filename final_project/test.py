from matplotlib import pyplot as plt
import pickle
import sys

sys.path.append('../tools/')
from feature_format import featureFormat, targetFeatureSplit

features_list = ['poi', 'salary', 'from_poi_to_this_person', 'from_this_person_to_poi', 'bonus', 'total_stock_value']
data = pickle.load(open('final_project_dataset.pkl', 'rb'))
# print(data[list(data.keys())[0]])
data = list(featureFormat(data, features_list, sort_keys=True))
factor = [0, 5, 4, 8, 4, 5]
# print(len(data))
for i in range(1, 6):
    for j in range(factor[i]):
        temp = [x[i] for x in data]
        idx = temp.index(max(temp))
        del data[idx]
# print(len(data))
labels, features = targetFeatureSplit(data)

for i in range(1, 6):
    for j in data:
        x = j[0]
        y = j[i]
        plt.scatter(x, y)
    plt.xlabel(features_list[0])
    plt.ylabel(features_list[i])
    plt.show()
