# -*- coding: utf-8 -*-
"""

@author: Ishan Bhattacharya

"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle

dataset = pd.read_csv("train.tsv", sep='\t')

dataset = dataset.iloc[:, 2:]
dataset = dataset[dataset.Sentiment != 2]

X = dataset.iloc[0:20000, 0].astype(str)
y = dataset.iloc[0:20000, 1].astype(str)

vectorizer = CountVectorizer(binary=True)
X_vectors = vectorizer.fit_transform(X)

print(vectorizer.get_feature_names())
print(X_vectors.toarray())

from sklearn import svm
svm_clf = svm.SVC(kernel='rbf')
svm_clf.fit(X_vectors, y)

pickle.dump(svm_clf, open('model.pkl','wb'))
pickle.dump(vectorizer, open('preprocessing.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
preprocessing = pickle.load(open('preprocessing.pkl','rb'))
test_x_vectors2 = preprocessing.transform(["bad"])
print(model.predict(test_x_vectors2))