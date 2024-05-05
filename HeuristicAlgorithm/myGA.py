#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987


import numpy as np
from sko.base import SkoBase
from sko.tools import func_transformer
from abc import ABCMeta, abstractmethod
from sko.operators import crossover, mutation, ranking, selection


class GeneticAlgorithmBase(SkoBase, metaclass=ABCMeta):
    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200, prob_mut=0.001,
                 constraint_eq=tuple(), constraint_ueq=tuple(), early_stop=None):
        self.func = func_transformer(func)
        assert size_pop % 2 == 0, 'size_pop must be even integer'
        self.size_pop = size_pop  # size of population
        self.max_iter = max_iter
        self.prob_mut = prob_mut  # probability of mutation
        self.n_dim = n_dim
        self.early_stop = early_stop

        # constraint:
        self.has_constraint = len(constraint_eq) > 0 or len(constraint_ueq) > 0
        self.constraint_eq = list(constraint_eq)  # a list of equal functions with ceq[i] = 0
        self.constraint_ueq = list(constraint_ueq)  # a list of unequal constraint functions with c[i] <= 0

        self.Chrom = None
        self.X = None  # shape = (size_pop, n_dim)
        self.Y_raw = None  # shape = (size_pop,) , value is f(x)
        self.Y = None  # shape = (size_pop,) , value is f(x) + penalty for constraint
        self.FitV = None  # shape = (size_pop,)

        # self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_Y = []

        self.all_history_Y = []
        self.all_history_FitV = []

        self.best_x, self.best_y = None, None

    @abstractmethod
    def chrom2x(self, Chrom):
        pass

    def x2y(self):
        self.Y_raw = self.func(self.X)
        if not self.has_constraint:
            self.Y = self.Y_raw
        else:
            # constraint
            penalty_eq = np.array([np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X])
            penalty_ueq = np.array([np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq])) for x in self.X])
            self.Y = self.Y_raw + 1e5 * penalty_eq + 1e5 * penalty_ueq
        return self.Y

    @abstractmethod
    def ranking(self):
        pass

    @abstractmethod
    def selection(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    @abstractmethod
    def mutation(self):
        pass

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        best = []
        for i in range(self.max_iter):
            self.X = self.chrom2x(self.Chrom)
            self.Y = self.x2y()
            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)

            if self.early_stop:
                best.append(min(self.generation_best_Y))
                if len(best) >= self.early_stop:
                    if best.count(min(best)) == len(best):
                        break
                    else:
                        best.pop(0)

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y

    fit = run


class GA(GeneticAlgorithmBase):
    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200,
                 prob_mut=0.001,
                 lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple(),
                 precision=1e-7, early_stop=None,
                 services=5, nodes=5, first_chrom=None):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, constraint_eq, constraint_ueq, early_stop)

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        self.precision = np.array(precision) * np.ones(self.n_dim)  # works when precision is int, float, list or array
        self.len_chrom = 8 * self.n_dim
        self.services = services
        self.nodes = nodes
        self.first_chrom = first_chrom

        self.crtbp()

    def crtbp(self):
        # 将输入进来的first_chrom转换为二进制编码
        if self.first_chrom is not None:
            self.Chrom = np.random.randint(low=0, high=2, size=(self.size_pop, self.len_chrom))
            for i in range(self.n_dim):
                start = i * 8
                end = (i + 1) * 8
                self.Chrom[0, start:end] = [int(c) for c in self.encode_decimal(self.first_chrom[i], 8)]
        else:
            self.Chrom = np.random.randint(low=0, high=2, size=(self.size_pop, self.len_chrom))
        return self.Chrom

    def decode_binary(self, binary_string):
        return int(binary_string, 2)

    def encode_binary(self, decimal_number, length):
        binary_string = bin(decimal_number)[2:]
        padding_length = max(0, length - len(binary_string))
        return '0' * padding_length + binary_string

    def encode_decimal(self, decimal_number, length, scale=100):
        scaled_number = int(decimal_number * scale)
        return self.encode_binary(scaled_number, length)

    def decode_decimal(self, encoded_binary, scale=100):
        decoded_number = self.decode_binary(encoded_binary)
        return decoded_number / scale

    def chrom2x(self, Chrom):
        X = np.zeros(shape=(self.size_pop, self.n_dim))
        for i in range(self.size_pop):
            for j in range(self.n_dim):
                start = j * 8
                end = (j + 1) * 8
                X[i, j] = self.decode_decimal(''.join([str(c) for c in Chrom[i, start:end]]))

        X = np.clip(X, 0, 1)
        for i in range(self.size_pop):
            X[i][:self.services * self.nodes] = np.round(X[i][:self.services * self.nodes], decimals=0)
        return X

    ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover.crossover_2point_bit
    mutation = mutation.mutation
