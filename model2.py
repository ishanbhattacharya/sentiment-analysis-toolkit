# -*- coding: utf-8 -*-
"""

@author: Ishan Bhattacharya

"""

import pandas as pd
import pickle
import spacy

nlp = spacy.load("en_core_web_md")

dataset = pd.read_csv("train.tsv", sep='\t')

dataset = dataset.iloc[:, 2:]
dataset = dataset[dataset.Sentiment != 2]

X = dataset.iloc[0:20000, 0].astype(str)
y = dataset.iloc[0:20000, 1].astype(str)

X_list = X.tolist()

docs = [nlp(x) for x in X_list]

from sklearn import svm
svm_wv = svm.SVC(kernel='rbf')
svm_wv.fit([x.vector for x in docs], y)

pickle.dump(svm_wv, open('model2.pkl','wb'))

model = pickle.load(open('model2.pkl','rb'))
test_x_vectors2 = [x.vector for x in [engine("bad")]]
print(model.predict(test_x_vectors2))