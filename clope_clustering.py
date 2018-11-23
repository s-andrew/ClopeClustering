from itertools import chain, repeat

import numpy as np
from scipy import sparse

class Clope:
    def __init__(self, r=2.6, n_iter=3):
        self.r = r
        self.n_iter = n_iter
        self.n_objects = None
        self.n_transactions = None
        self.vectorize_cost = np.vectorize(self.cost)

    def _init(self):
        self.clusters = sparse.csr_matrix((1, self.n_objects), dtype='uint8')
        self.t_matrix = sparse.csc_matrix((self.n_transactions, 1), dtype='uint8')

    def add_empty_cluster(self):
        self.clusters = sparse.vstack([self.clusters, sparse.csr_matrix((1, self.n_objects), dtype='uint8')])
        self.t_matrix = sparse.hstack([self.t_matrix, sparse.csc_matrix((self.n_transactions, 1), dtype='uint8')])

    def initialization(self, X):
        self._init()
        for i, transaction in enumerate(X):
            pass

    def add_costs(self, transaction):
        s = np.ravel(self.clusters.sum(axis=1))
        n = self.t_matrix.getnnz(axis=0)
        w = self.clusters.getnnz(axis=1)

        s_new = s + transaction.sum()
        n_new = n + 1
        w_new = np.ravel(np.count_nonzero(self.clusters + transaction, axis=1))

        return self.vectorize_cost(s_new, n_new, w_new) - self.vectorize_cost(s, n, w)

    def remove_cost(self, cluster_index, transaction):
        s = self.clusters[cluster_index].sum()
        n = self.t_matrix[:, cluster_index].sum()
        w = self.clusters[cluster_index].getnnz()

        s_new = s - transaction.sum()
        n_new = n - 1
        w_new = (self.clusters[cluster_index] - transaction).getnnz()

        return self.cost(s_new, n_new, w_new) - self.cost(s, n, w)

    def cost(self, s, n, w):
        if s == 0 or n == 0 or w == 0:
            return 0
        return s * n / (w ** self.r)

    def get_current_cluster(self, transaction_index):
        _, nonzero = self.t_matrix[transaction_index].nonzero()
        if nonzero.shape != (1,):
            raise ValueError('transaction include in many clusters')
        return nonzero[0]

    def add_transaction(self, X, cluster_index, transaction_index):
        self.clusters[cluster_index] += X[transaction_index]
        self.t_matrix[transaction_index, cluster_index] = 1

    def remove_transaction(self, X, cluster_index, transaction_index):
        self.clusters[cluster_index] -= X[transaction_index]
        self.t_matrix[transaction_index, cluster_index] = 0







if __name__ == '__main__':
    pass
