from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pickle5 as pkl
import numpy as np
import torch

def load_data(dataset):
    train = load_dataset(dataset, split='train').to_pandas()
    dev = load_dataset(dataset, split='validation').to_pandas()
    test = load_dataset(dataset, split='test').to_pandas()
    return train, dev, test


def find_ner_x_y(hf_dataset, model_name, device=0):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=device)

    y = []
    text = []
    dict_ner = {}
    for i in range(len(hf_dataset)):
        a = TreebankWordDetokenizer().detokenize(hf_dataset['tokens'][i])
        text += [a]
        ner = nlp(a)
        dict_ner[i] = ner
        if not ner:
            y += [0]
        else:
            y += [1]
    return text, y, dict_ner

def find_x_y_emotion(hf_dataset, device=0):
    classifier = pipeline("text-classification",model='bhadresh-savani/bert-base-uncased-emotion', device=device,  return_all_scores=True)
    texts = hf_dataset.text.to_numpy()
    labels = []
    for text in texts:
        score = []
        prediction = classifier(text)
        for counter in range(6):
            score += [prediction[0][counter]['score']]
        labels += [np.argmax(np.asarray(score))]

    return texts, np.asarray(labels)

def find_x_y_sentiment(texts,device=0):
    model = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=model, device=device, return_all_scores=True)
    labels = []
    final_texts = []
    for each_text in texts:
        text = preprocess(each_text)
        score = []
        prediction = classifier(text)
        for counter in range(3):
            score += [prediction[0][counter]['score']]
        labels += [np.argmax(np.asarray(score))]
        final_texts.append(text)

    return np.asarray(final_texts), np.asarray(labels)

def save_pkl(filename, text, y, dict_ner=[]):
    f = open(filename,'wb')
    dump = {}
    dump['labels'] = y
    dump['ner'] = dict_ner
    dump['text'] = text
    pkl.dump(dump, f, protocol=pkl.HIGHEST_PROTOCOL)
    f.close()

def open_pkl(filename):
    with open(filename, 'rb') as handle:
        dump = pkl.load(handle)
    y = np.asarray(dump['labels'])
    dict_ner = dump['ner']
    text = dump['text']
    return text, y, dict_ner

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)