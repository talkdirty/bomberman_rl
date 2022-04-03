import os
import pickle
import numpy as np
import lzma
import bombergym.settings as s

import xgboost as xgb
from sklearn.model_selection import train_test_split

class GameplayDataset():
    def __init__(self, directory):
        self.directory = directory
        self.files = os.listdir(directory)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        try:
            with lzma.open(f'{self.directory}/{self.files[item]}') as fd:
                state, action = pickle.load(fd)[:2]
        except EOFError:
            print(f'Warn: {item}, {self.files[item]} is broken.')
            return random.choice(self)
        return state, action 

if __name__ == '__main__':
    if not os.path.exists('xgb_test.buffer'):
        print('Loading data from scratch')
        ds = GameplayDataset('out_220403_v3_reduced')
        X = np.zeros((len(ds), 25))
        y = np.zeros(len(ds))
        for i in range(len(ds)):
            X[i, :] = ds[i][0].flatten()
            y[i] = ds[i][1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        dtest.save_binary('xgb_test.buffer')
        dtrain.save_binary('xgb_train.buffer')
    else:
        print('Loading data from xgb buffer')
        dtrain = xgb.DMatrix('xgb_train.buffer')
        dtest = xgb.DMatrix('xgb_test.buffer')
    print('Done loading')
    param = {'max_depth':9, 'eta':1, 'objective':'multi:softprob', 'num_class':6 , 'verbosity': 2}
    num_round = 2
    res = dict()
    bst = xgb.train(param, dtrain, num_round, evals_result=res)
    # make prediction
    preds = bst.predict(dtest)
    # err = np.count_nonzero(dtest.get_label() - preds) / len(preds)
    # print(f'Err: {err:.3f}')
    bst.save_model('bst0001.model')
    print(f'Dumping booster to bst0001.model')
