import sys
sys.path.append('../src/')
import util_funcs as uf


with open('../data/sentiment/raw_data/test_text.txt') as f:
    test = f.readlines()
with open('../data/sentiment/raw_data/train_text.txt') as f:
    train = f.readlines()


text_train , y_train = uf.find_x_y_sentiment(train)
uf.save_pkl('../data/sentiment/train_sentiment.pkl', text_train, y_train)

text_test , y_test = uf.find_x_y_sentiment(test)
uf.save_pkl('../data/sentiment/test_sentiment.pkl', text_test, y_test)