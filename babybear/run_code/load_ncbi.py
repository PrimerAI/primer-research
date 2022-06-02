import sys
sys.path.append('../src/')
import util_funcs as uf

dataset = 'ncbi_disease'
model_name = "fidukm34/biobert_v1.1_pubmed-finetuned-ner-finetuned-ner"
train, dev, test = uf.load_data(dataset)

text_train , y_train, dict_ner_train = uf.find_ner_x_y(train, model_name)
uf.save_pkl('../data/ncbi_disease/train_ncbi.pkl', text_train, y_train, dict_ner_train)

text_test , y_test, dict_ner_test = uf.find_ner_x_y(test, model_name)
uf.save_pkl('../data/ncbi_disease/test_ncbi.pkl', text_test, y_test, dict_ner_test)