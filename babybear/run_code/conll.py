import sys
sys.path.append('../src/')

# from primer_nlx.embedding.SIF import SIF

import numpy as np
import pickle5 as pkl
import tensorflow_hub as hub
import util_funcs as uf
from nlx_babybear import RFBabyBear
from inference_triage import MamabearClassifier, TriagedClassifier
from transformers import AutoTokenizer, AutoModelForTokenClassification

from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt



filename = '../data/conll2003/train_conll.pkl'
texts_train, y_train, _ = uf.open_pkl(filename)
doc, labels = np.asarray(texts_train), np.asarray(y_train)

filename = '../data/conll2003/test_conll.pkl'
texts_test, y_test, _ = uf.open_pkl(filename)
texts_test, y_test = np.asarray(texts_test), np.asarray(y_test)

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
confidence_th_options = np.arange(0,1.005,.005)
metric = "accuracy"
metric_threshold = .99

language_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
mamabear = MamabearClassifier(model, tokenizer=tokenizer)
babybear = RFBabyBear(language_model)

inf_traige = TriagedClassifier("ner", babybear, mamabear, metric_threshold, "accuracy", confidence_th_options)

inf_traige.train(doc, labels)

print(f"Confidence threshold is: {inf_traige.confidence_th}")

print(f"The following plots are the saving vs Threshold for different CV fold")

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

with open('../output/conll.resullts', 'wb') as outp:  # Overwrites any existing file.
        pkl.dump(dump_data, outp, pkl.HIGHEST_PROTOCOL)
