import sys
sys.path.append('../src/')

# from primer_nlx.embedding.SIF import SIF

import numpy as np
import pickle5 as pkl
import tensorflow_hub as hub
import util_funcs as uf
from nlx_babybear import RFBabyBear
from inference_triage import PapabearClassifierSentiment, TriagedClassifier

from transformers import AutoTokenizer, AutoModelForTokenClassification

from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


filename = '../data/sentiment/train_sentiment.pkl'
texts_train, y_train, _ = uf.open_pkl(filename)
doc, labels = np.asarray(texts_train[0:1000]), np.asarray(y_train[0:1000])


filename = '../data/sentiment/test_sentiment.pkl'
texts_test, y_test, _ = uf.open_pkl(filename)
texts_test, y_test = np.asarray(texts_test[0:500]), np.asarray(y_test[0:500])

model='cardiffnlp/twitter-roberta-base-sentiment-latest'
metric = "accuracy"
metric_threshold = .9
confidence_th_options = np.arange(0,1.005,.005)

language_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
# language_model = SIF("latest-small")
papabear = PapabearClassifierSentiment(model)
babybear = RFBabyBear(language_model)

inf_traige = TriagedClassifier("classification", babybear, papabear, metric_threshold, "accuracy", confidence_th_options)

inf_traige.train(doc, labels)

print(f"Confidence threshold is: {inf_traige.confidence_th}")

print(f"The following plots are the saving vs Threshold for different CV fold")

babybear = RFBabyBear(language_model)
babybear.train(doc, labels, n_class=len(np.unique(labels)))

inf_traige.babybear = babybear
a = inf_traige.score(texts_test, y_test)

dump_data = {}
dump_data['result'] = a
dump_data['confidence_th'] = inf_traige.confidence_th
dump_data['indx_conf_th'] = inf_traige.indx_conf_th
dump_data['metric'] = inf_traige.metric
dump_data['metric_threshold'] = inf_traige.metric_threshold
dump_data['performance'] = inf_traige.performance
dump_data['saving'] = inf_traige.saving
dump_data['tot_time'] = inf_traige.tot_time

with open('../output/sentiment.resullts', 'wb') as outp:  # Overwrites any existing file.
        pkl.dump(dump_data, outp, pkl.HIGHEST_PROTOCOL)
