import os
import pickle
import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

sys.path.append("../tools/")
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

from_sara = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

# count = 0

# temp_counter is a way to speed up the development--there are
# thousands of emails from Sara and Chris, so running over all of them
# can take a long time
# temp_counter helps you only look at the first 200 emails in the list so you
# can iterate your modifications quicker
stop_words = set(ENGLISH_STOP_WORDS)
for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        path = os.path.join('..', path[:-1])
        print(path)
        email = open(path, "r")
        data = parseOutText(email)
        names = ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf", "germany"]
        data = data.split()
        final = []
        for i in data:
            if i not in names:
                flag = 0
                value = ''
                for j in names:
                    if i.startswith(j) or i.endswith(j):
                        flag += 1
                        value = j
                if flag == 0:
                    final.append(i)
                else:
                    final.append(i.replace(value, ''))
        word_data.append(' '.join(final))
        # append a 0 to from_data if email is from Sara, and 1 if email is from Chris
        from_data.append(0) if name == 'sara' else from_data.append(1)
        email.close()

print("emails processed")
from_sara.close()
from_chris.close()

pickle.dump(word_data, open("your_word_data.pkl", "wb"))
pickle.dump(from_data, open("your_email_authors.pkl", "wb"))
# in Part 4, do TfIdf vectorization here
vectorizer = TfidfVectorizer(stop_words=stop_words)
x = vectorizer.fit_transform(word_data)
