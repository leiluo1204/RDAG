import pickle
import numpy as np

# f = open('./data/DailyDialog/label_vocab.pkl','rb')
f = open('./data/DailyDialog/speaker_vocab.pkl','rb')
data = pickle.load(f)
print(data)