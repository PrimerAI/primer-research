
import sys
sys.path.append('../src/')
import util_funcs as uf

dataset = 'conll2003'
model_name = "dslim/bert-base-NER"
train, dev, test = uf.load_data(dataset)

#text_train , y_train, dict_ner_train = uf.find_ner_x_y(train, model_name)
#uf.save_pkl('../data/conll2003/train_conll.pkl', text_train, y_train, dict_ner_train)

text_test , y_test, dict_ner_test = uf.find_ner_x_y(test, model_name)
uf.save_pkl('../data/conll2003/test_conll.pkl', text_test, y_test, dict_ner_test)
