# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np
import six
import pickle
import scipy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from tqdm import tqdm
import scipy.stats as ss
from sklearn.preprocessing import StandardScaler
from learning2rank.utils import plot_result
from learning2rank.utils import NNfuncs

######################################################################################
# Define model
class Model(chainer.Chain):
    """
    RankNet - Pairwise comparison of ranking.
    The original paper:
        http://research.microsoft.com/en-us/um/people/cburges/papers/ICML_ranking.pdf
    Japanese only:
        http://qiita.com/sz_dr/items/0e50120318527a928407
    """
    def __init__(self, n_in, n_units1, n_units2, n_out):
        super(Model, self).__init__(
            l1=L.Linear(n_in, n_units1),
            l2=L.Linear(n_units1, n_units2),
            l3=L.Linear(n_units2, n_out),
        )
    def __call__(self, x_i, x_j, t_i, t_j):
        s_i = self.l3(F.relu(self.l2(F.relu(self.l1(x_i)))))
        s_j = self.l3(F.relu(self.l2(F.relu(self.l1(x_j)))))
        s_diff = s_i - s_j
        if t_i.data > t_j.data:
            S_ij = 1
        elif t_i.data < t_j.data:
            S_ij = -1
        else:
            S_ij = 0
        self.loss = (1 - S_ij) * s_diff / 2. + F.math.exponential.Log()(1 + F.math.exponential.Exp()(-s_diff))
        return self.loss

    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h = F.relu(self.l3(h2))
        return h.data


class RankNet(NNfuncs.NN):
    """
    RankNet training function.
    Usage (Initialize):
        RankModel = RankNet()

    Usage (Traininng):
        Model.fit(X, y)

    With options:
        Model.fit(X, y, batchsize=100, n_iter=5000, n_units1=512, n_units2=128, tv_ratio=0.95, optimizerAlgorithm="Adam", savefigName="result.pdf", savemodelName="RankNet.model")

    """
    def __init__(self, resumemodelName=None):
        self.resumemodelName = resumemodelName
        self.train_loss, self.test_loss = [], []
        if resumemodelName is not None:
            self.model = Model(4096, 2048, 128, 1)
            self.optimizer = optimizers.Adam()
            self.optimizer.setup(self.model)
            print("load resume model!")
            self.loadModel(resumemodelName)

    # Evaluation function of NDCG@100
    def ndcg(self, y_true, y_score, k=100):
        y_true = y_true.ravel()
        y_score = y_score.ravel()
        y_true_sorted = sorted(y_true, reverse=True)
        k = min(k,len(y_true_sorted))
        ideal_dcg = 0
        for i in range(k):
            ideal_dcg += (2 ** y_true_sorted[i] - 1.) / np.log2(i + 2)
        dcg = 0
        argsort_indices = np.argsort(y_score)[::-1]
        for i in range(k):
            dcg += (2 ** y_true[argsort_indices[i]] - 1.) / np.log2(i + 2)
        ndcg = dcg / ideal_dcg
        return ndcg

    # Training function
    def trainModel(self, x_train, y_train, x_test, y_test, n_iter):
        sigma = 5.0
        loss_step = 100

        for step in tqdm(range(n_iter)):
            i, j = np.random.randint(len(x_train), size=2)
            x_i = chainer.Variable(x_train[i].reshape(1, -1))
            x_j = chainer.Variable(x_train[j].reshape(1, -1))
            y_i = chainer.Variable(y_train[i])
            y_j = chainer.Variable(y_train[j])
            self.optimizer.update(self.model, x_i, x_j, y_i, y_j)

            if (step + 1) % loss_step == 0:
                train_score = self.model.predict(chainer.Variable(x_train))
                test_score = self.model.predict(chainer.Variable(x_test))
                # print("test score:")
                # print(test_score)
                # print("y_test:")
                # print(y_test)
                train_ndcg = self.ndcg(y_train, train_score)
                test_ndcg = self.ndcg(y_test, test_score)
                self.train_loss.append(train_ndcg)
                self.test_loss.append(test_ndcg)
                print("step: {0}".format(step + 1))
                print("NDCG@100 | train: {0}, test: {1}".format(train_ndcg, test_ndcg))

    def fit(self, fit_X, fit_y, batchsize=100, n_iter=5000, n_units1=512, n_units2=128, tv_ratio=0.95, optimizerAlgorithm="Adam", savefigName="result.pdf", savemodelName="RankNet.model"):
        train_X, train_y, validate_X, validate_y = self.splitData(fit_X, fit_y, tv_ratio)
        print("The number of data, train:", len(train_X), "validate:", len(validate_X))

        if self.resumemodelName is None:
            self.initializeModel(Model, train_X, n_units1, n_units2, optimizerAlgorithm)

        self.trainModel(train_X, train_y, validate_X, validate_y, n_iter)

        plot_result.acc(self.train_loss, self.test_loss, savename=savefigName)
        self.saveModels(savemodelName)

    def RankNetpredict(self, X, batchsize=100):
        return self.predict(X, batchsize)



################################################################################################
## end of file ##
################################################################################################


##################
## MEMO
#
# # loss func of ranknet
# def ndcg(y_true, y_score, k=100):
#     y_true = y_true.ravel()
#     y_score = y_score.ravel()
#     y_true_sorted = sorted(y_true, reverse=True)
#     ideal_dcg = 0
#     for i in range(k):
#         ideal_dcg += (2 ** y_true_sorted[i] - 1.) / np.log2(i + 2)
#     dcg = 0
#     argsort_indices = np.argsort(y_score)[::-1]
#     for i in range(k):
#         dcg += (2 ** y_true[argsort_indices[i]] - 1.) / np.log2(i + 2)
#     ndcg = dcg / ideal_dcg
#     return ndcg

# ######################################################################################
# # training func of ranknet
# def trainRankNet(model, optimizer, x_train, y_train, x_test, y_test, n_iter, N_train, N_test):
#     sigma = 5.0
#     loss_step = 100
#     train_ndcgs, test_ndcgs = [], []

#     for step in tqdm(range(n_iter)):
#         i, j = np.random.randint(N_train, size=2)
#         x_i = chainer.Variable(x_train[i].reshape(1, -1))
#         x_j = chainer.Variable(x_train[j].reshape(1, -1))
#         y_i = chainer.Variable(y_train[i])
#         y_j = chainer.Variable(y_train[j])
#         optimizer.update(model, x_i, x_j, y_i, y_j)

#         if (step + 1) % loss_step == 0:
#             train_score = model.predict(chainer.Variable(x_train))
#             test_score = model.predict(chainer.Variable(x_test))
#             train_ndcg = ndcg(y_train, train_score)
#             test_ndcg = ndcg(y_test, test_score)
#             train_ndcgs.append(train_ndcg)
#             test_ndcgs.append(test_ndcg)
#             print("step: {0}".format(step + 1))
#             print("NDCG@100 | train: {0}, test: {1}".format(train_ndcg, test_ndcg))
#     return model, optimizer, train_ndcg, test_ndcg


# ################################################################################################
# ## 学習部分 ##
# ################################################################################################

# # n_iter は、多分めちゃめちゃ多い方がよいと思う。イメージ、データ数くらい？

# def fit(fit_X, fit_y, batchsize=100, n_iter=5000, n_units1=512, n_units2=128, tv_ratio=0.95, optimizerAlgorithm="Adam", savefigName="result.png", savemodelName="Spearman.model", resumemodelName=None):
#     print('load dataset')
#     perm = np.random.permutation(len(fit_X))

#     N_train = np.floor(len(fit_X) * tv_ratio)
#     train_X, validate_X = np.split(fit_X[perm].astype(np.float32),   [N_train])
#     train_y, validate_y = np.split(fit_Y[perm].astype(np.float32).reshape(len(fit_Y), 1), [N_train])
#     N_validate = len(validate_y)
#     print("The number of data, train:",N_train, "validate:", N_validate)                # トレーニングとテストのデータ数を表示

#     if resumemodelName is None:
#         print("prepare initialized model!")
#         model = Model(len(x_train[0]), n_units1, n_units2, 1)
#         if optimizerAlgorithm == "Adam":
#             optimizer = optimizers.Adam()
#         elif optimizerAlgorithm == "AdaGrad":
#             optimizer = optimizers.AdaGrad()
#         elif optimizerAlgorithm == "SGD":
#             optimizer = optimizers.MomentumSGD()
#         else:
#             raise ValueError('could not find %s in optimizers {"Adam", "AdaGrad", "SGD"}' % (optimizerAlgorithm))
#         optimizer.setup(model)
#     else:
#         print("load resume model!")
#         model, optimizer =  loadModel(resumemodelName)

#     model, optimizer, train_loss, test_loss = trainRankNet(model, optimizer, train_X, train_y, validate_X, validate_y, n_iter, N_train, N_validate)

# #    plot_result(train_loss, test_loss, savename=savefigName)
#     print('save the model')
#     serializers.save_hdf5(savemodelName, model)
#     print('save the optimizer')
#     serializers.save_hdf5(savemodelName[:-5]+ 'state', optimizer)
#     return model


# ################################################################################################
# ## 予測 ##
# ################################################################################################

# def predict(model, predict_X, batchsize=100):
#     predict_y = predictTargets(model, predict_X.astype(np.float32), batchsize)
#     return predict_y

# def loadModel(modelName):
#     print('Load model')
#     serializers.load_hdf5(modelName, model)
#     print('Load optimizer state')
#     serializers.load_hdf5(modelName[:-5] + 'state', optimizer)
#     return model, optimizer
