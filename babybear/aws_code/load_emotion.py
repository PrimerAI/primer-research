
import sys
sys.path.append('../src/')
import util_funcs as uf

dataset = 'emotion'
train, dev, test = uf.load_data(dataset)

text_train , y_train = uf.find_x_y_emotion(train)
uf.save_pkl('../data/emotion/train_emotion.pkl', text_train, y_train)

text_test , y_test = uf.find_x_y_emotion(test)
uf.save_pkl('../data/emotion/test_emotion.pkl', text_test, y_test)