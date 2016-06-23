# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np
import six
from tqdm import tqdm
import chainer

######################################################################################
# モデルを使って予測する関数
def predictTargets(model, x_pred, batchsize):
    N_pred = len(x_pred)
    y_pred = np.zeros(0)
    for j in tqdm(six.moves.range(0, N_pred, batchsize)):
        x = chainer.Variable(np.asarray(x_pred[j:j + batchsize]), volatile='on')
        y_pred = np.append(y_pred, model.predict(x))
    return y_pred
