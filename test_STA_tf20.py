# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Bidirectional, Concatenate, Dot, Input, LSTM
from tensorflow.keras.layers import RepeatVector, Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K

import numpy as np
from sklearn import preprocessing
import pandas as pd
from tqdm import tqdm
from gc import collect
from memory_profiler import profile
import line_profiler
import os
import random


# 建立seq to seq 模型
class Buliding_Model():
    def model(self, Tx, Ty, n_a, n_s, learn_rate=0.1):
        def softmax(x, axis=1):
            ndim = K.ndim(x)
            if ndim == 2:
                return K.softmax(x)
            elif ndim > 2:
                e = K.exp(x - K.max(x, axis=axis, keepdims=True))
                s = K.sum(e, axis=axis, keepdims=True)
                return e / s
            else:
                raise ValueError('Cannot apply softmax to a tensor that is 1D')

        X = Input(shape=(Tx, 1))
        s0 = Input(shape=(n_s,), name='s0')
        c0 = Input(shape=(n_s,), name='c0')
        s = s0
        c = c0

        outputs = []

        a = Bidirectional(LSTM(n_a, return_sequences=True))(X)

        repeator = RepeatVector(Tx)
        concatenator = Concatenate(axis=-1)
        densor1 = Dense(100, activation="relu")
        densor2 = Dense(1, activation="relu")
        activator = Activation(softmax,
                               name='attention_weights')  # We are using a custom softmax(axis = 1) loaded in this notebook
        dotor = Dot(axes=1)
        post_activation_LSTM_cell = LSTM(n_s, return_state=True)
        output_layer = Dense(1)

        def one_step_attention(a, s_prev):

            s_prev = repeator(s_prev)

            concat = concatenator([a, s_prev])
            e = densor1(concat)
            energies = densor2(e)
            alphas = activator(energies)
            context = dotor([alphas, a])
            return context

        for t in range(Ty):
            context = one_step_attention(a, s)

            s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

            out = output_layer(s)
            outputs.append(out)

        model = Model(inputs=[X, s0, c0], outputs=outputs)
        opt = Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999)
        model.compile(loss='mse', optimizer=opt, metrics=['mse'])

        return model

    ### K-折交叉验证函数分割训练集和测试集


def train_test_split(X_train, y_train, cv):
    size = int(X_train.shape[1] / cv)
    K = [i * size for i in range(cv + 1)]
    K = sorted(K, reverse=True)

    Kfold = []
    for i in range(cv):
        temp = []
        temp = K[i:i + 2]
        Kfold.append(temp)

    X_train_set = []
    y_train_set = []
    for i in range(cv):
        X_test = X_train[:, Kfold[i][1]:Kfold[i][0]]  #
        y_test = y_train[:, Kfold[i][1]:Kfold[i][0]]  #
        X_train_temp = []
        y_train_temp = []
        for j in range(cv):
            if j != i:
                X_train_temp.append(X_train[:, Kfold[j][1]:Kfold[j][0]])  #
                y_train_temp.append(y_train[:, Kfold[j][1]:Kfold[j][0]])  #
        X_train_set_sub = np.concatenate((X_train_temp), axis=-1)
        y_train_set_sub = np.concatenate((y_train_temp), axis=-1)
        X_train_set.append([X_train_set_sub, X_test])
        y_train_set.append([y_train_set_sub, y_test])

    return X_train_set, y_train_set


df = pd.read_csv(os.path.join(os.sys.path[0], 'new_selected_data.csv'), header=None)
X_train = df.iloc[:161, :].values
y_train = df.iloc[161:, :].values

X_train_set, y_train_set = train_test_split(X_train, y_train, 7)


@profile
def fitness(tree):
    y_test_pre_set = []
    # history_set = []
    for k in tqdm(range(7), desc='模型训练'):
        model = Buliding_Model()

        X_train_test, X_test_test = X_train_set[k]
        y_train_test, y_test_test = y_train_set[k]

        scaler_xtrain = preprocessing.StandardScaler().fit(X_train_test)
        scaler_xtest = preprocessing.StandardScaler().fit(X_test_test)
        scaler_ytrain = preprocessing.StandardScaler().fit(y_train_test)
        scaler_ytest = preprocessing.StandardScaler().fit(y_test_test)

        X_train_std = scaler_xtrain.transform(X_train_test).T
        X_test_std = scaler_xtest.transform(X_test_test).T
        y_train_std = scaler_ytrain.transform(y_train_test).T
        y_test_std = scaler_ytest.transform(y_test_test).T

        Xo_train = X_train_std.reshape([-1, 161, 1])
        Xo_test = X_test_std.reshape([-1, 161, 1])
        Yo_train = y_train_std.reshape([-1, 7, 1])

        Tx = 161
        Ty = 7
        n_a = 32  # number of cell
        n_s = 64
        m = X_train_std.shape[0]  # number of port
        s0 = np.zeros((m, n_s))
        c0 = np.zeros((m, n_s))
        outputs = list(Yo_train.swapaxes(0, 1))

        model = model.model(Tx, Ty, n_a, n_s, learn_rate=tree)
        model.fit([Xo_train, s0, c0], outputs, epochs=1, batch_size=100, verbose=0)
        # history_set.append(sum(history.history['loss']))
        m_t = X_test_std.shape[0]
        s0_t = np.zeros((m_t, n_s))
        c0_t = np.zeros((m_t, n_s))
        yo_p = model.predict([Xo_test, s0_t, c0_t])
        yo_p = np.array(yo_p)
        yo_p = yo_p.swapaxes(0, 1)
        yo_p = yo_p.reshape([-1, 7])
        y_test_pr = scaler_ytest.inverse_transform(yo_p.T)
        y_test_pre_set.append(y_test_pr)
        # print('损失函数:',history.history['loss'])

    # np.save(r'/home/hbut/PycharmProjects/untitled/y_test_pre_set_50.npy',y_test_pre_set)

    y_test_ture_set = []
    for k in range(7):
        y_test = y_train_set[k][1]
        y_test_ture_set.append(y_test)

    # np.save(r'/home/hbut/PycharmProjects/untitled/y_test_ture_set_50.npy',y_test_ture_set)

    y_test_pre_set = np.array(y_test_pre_set)
    y_test_ture_set = np.array(y_test_ture_set)

    MAPE = np.sum(
        np.sum(np.sum(np.abs(y_test_pre_set - y_test_ture_set) / y_test_ture_set, axis=0) / 7, axis=-1) / 2863) / 7
    print('百分比误差', MAPE)
    # del y_test_pre_set, y_test_ture_set, model
    # gc.collect()

    return MAPE


class TSA(object):
    """docstring for TSA"""

    def __init__(self, n, d, st, termination):
        super(TSA, self).__init__()
        self.d = d
        self.n = n
        self.st = st
        self.termination = termination
        self.low = (n * 0.1)
        self.high = round(n * 0.25)
        self.fit = [0 for i in range(self.n)]
        self.stand = [[0 for i in range(self.d)] for i in range(self.n)]
        self.best_params = []

    def runTSA(self):
        self.initialize()
        indis = self.find_best()
        self.best_fit = self.fit[indis]
        self.best_params = self.stand[indis]

        fes_counter = self.n
        while (fes_counter < self.termination):
            for i in range(0, self.n):
                fes_counter += self.seed_production(i)
            indis = self.find_best()
            if self.fit[indis] < self.best_fit:
                self.best_fit = self.fit[indis]
                self.best_params = self.stand[indis]
            print("FES=" + str(fes_counter) + " The Best Found= " + str(self.best_fit))

    def initialize(self):
        for i in tqdm(range(0, self.n), desc='种群数量'):
            for j in range(0, self.d):
                self.stand[i][j] = random.random()
            self.fit[i] = self.objective(self.stand[i])
            collect()

    #     def objective(self, tree):
    #         result = 0
    #         if self.function == 1:
    #             for i in range(0, self.d):
    #                 result += tree[i] * tree[i]
    #         elif self.function == 2:
    #             for i in range(0, self.d):
    #                 result += math.pow(tree[i], 2) - 10 * math.cos(2 * math.pi * tree[i])
    #             result = 10 * d + result
    #         return result

    def objective(self, tree):
        result = 0
        for i in range(0, self.d):
            deviation_rate = fitness(tree[0])
        return deviation_rate

    def find_best(self):
        fit = self.fit[0]
        indis = 0
        for i in range(1, self.n):
            if fit < self.fit[i]:
                indis = i
                fit = self.fit[i]
        return indis

    def seed_production(self, tree_indis):
        i, j = 0, 0
        r = random.randint(0, self.n - 1)
        while r == tree_indis:
            r = random.randint(0, self.n - 1)
        seed_number = int(round(self.low + (self.high - self.low) * random.random()))
        seed_fit = [0 for i in range(seed_number)]
        seeds = [[0 for i in range(self.d)] for i in range(seed_number)]
        for i in range(0, seed_number):
            for j in range(0, self.d):
                alfa = (random.random() - 0.5) * 2
                if random.random() < self.st:
                    seeds[i][j] = self.stand[tree_indis][j] + (self.best_params[j] - self.stand[r][j]) * alfa
                else:
                    seeds[i][j] = self.stand[tree_indis][j] + (self.stand[tree_indis][j] - self.stand[r][j]) * alfa;
                if (seeds[i][j] > self.ub) or (seeds[i][j] < self.lb):
                    seeds[i][j] = self.lb + (self.ub - self.lb) * random.random()
            seed_fit[i] = self.objective(seeds[i])
        for i in range(0, seed_number):
            if seed_fit[i] < self.fit[tree_indis]:
                self.fit[tree_indis] = seed_fit[i]
                self.stand[tree_indis] = seeds[i]
        return seed_number


def main():
    n, st, d = 20, 0.1, 1
    termination = 30
    tsa = TSA(n, d, st, termination)
    tsa.runTSA()


if __name__ == '__main__':
    main()
