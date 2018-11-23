import unittest

import numpy as np
from scipy import sparse

from clope_clustering import Clope



class TestClopeClustering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.transactions = sparse.csr_matrix([[0, 1, 1, 1, 1],
                                              [0, 1, 0, 1, 0],
                                              [1, 0, 0, 0, 0],
                                              [0, 1, 1, 0, 0],
                                              [1, 0, 0, 1, 0],
                                              [0, 1, 1, 0, 1],
                                              [1, 0, 0, 1, 0],
                                              [1, 1, 0, 0, 1],
                                              [0, 0, 0, 0, 1],
                                              [1, 1, 0, 0, 1],
                                              [0, 1, 1, 1, 0],
                                              [0, 0, 0, 1, 1],
                                              [0, 0, 0, 1, 0],
                                              [1, 1, 0, 1, 1],
                                              [1, 1, 1, 0, 0],
                                              [0, 0, 1, 1, 0],
                                              [1, 1, 0, 0, 0],
                                              [1, 1, 1, 0, 0],
                                              [0, 0, 1, 0, 1],
                                              [1, 0, 0, 1, 1],
                                              [1, 1, 1, 0, 1],
                                              [1, 0, 0, 1, 0],
                                              [0, 0, 0, 0, 1],
                                              [1, 0, 1, 0, 1],
                                              [0, 0, 0, 1, 0],
                                              [1, 0, 1, 0, 0],
                                              [1, 0, 1, 0, 0],
                                              [0, 0, 1, 0, 0],
                                              [0, 1, 0, 1, 0],
                                              [1, 0, 0, 1, 0],
                                              [0, 1, 0, 0, 1],
                                              [0, 0, 0, 1, 1],
                                              [0, 1, 1, 0, 1],
                                              [0, 0, 1, 1, 1],
                                              [0, 1, 1, 1, 1],
                                              [0, 1, 0, 1, 0],
                                              [1, 0, 1, 1, 0],
                                              [1, 0, 0, 0, 0],
                                              [1, 0, 1, 1, 1],
                                              [0, 0, 0, 0, 0]], dtype='uint8')
        cls.n_transactions, cls.n_objects = cls.transactions.shape
        cls.clusters = sparse.csr_matrix([np.ravel(cls.transactions[i * 10: (i + 1) * 10].sum(axis=0)) for i in range(4)])
        cls.t_matrix = sparse.csc_matrix((cls.n_transactions, 4), dtype='uint8')
        k = 0
        for i in range(4):
            for j in range(10):
                cls.t_matrix[k + j, i] = 1
            k += 10

    def test_add_costs(self):
        clp = Clope()
        clp.n_transactions = self.n_transactions
        clp.n_objects = self.n_objects
        clp.clusters = self.clusters
        clp.t_matrix = self.t_matrix

        transaction = np.array([0, 0, 0, 0, 1], dtype='uint8')
        transaction = self.transactions[0]

        s = np.ravel(self.clusters.sum(axis=1))
        n = self.t_matrix.getnnz(axis=0)
        w = self.clusters.getnnz(axis=1)
        s_new = s + transaction.sum()
        n_new = n + 1
        w_new = np.ravel(np.count_nonzero(self.clusters + transaction, axis=1))

        costs1 = clp.vectorize_cost(s_new, n_new, w_new) - clp.vectorize_cost(s, n, w)
        costs2 = clp.add_costs(transaction)
        self.assertTrue(all(costs1 == costs2), 'PIZDEC')

    def test_remove_cost(self):
        clp = Clope()
        clp.n_transactions = self.n_transactions
        clp.n_objects = self.n_objects
        clp.clusters = self.clusters
        clp.t_matrix = self.t_matrix

        cluster_index = 0
        transaction = self.transactions[0]

        s = self.clusters[cluster_index].sum()
        n = self.t_matrix[:, cluster_index].sum()
        w = self.clusters[cluster_index].getnnz()

        s_new = s - transaction.sum()
        n_new = n - 1
        w_new = (self.clusters[cluster_index] - transaction).getnnz()

        cost1 = clp.cost(s_new, n_new, w_new) - clp.cost(s, n, w)
        cost2 = clp.remove_cost(cluster_index, transaction)

        self.assertEqual(cost1, cost2, 'PIZDEC')


        for i, transaction in enumerate(self.transactions):
            print('\n###', i, transaction)
            break



