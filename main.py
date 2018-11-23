import numpy as np
from scipy import sparse

if __name__ == '__main__':
    A = np.array([[1, 0, 0, 0, 1, 0],
                  [1, 1, 0, 1, 0, 0],
                  [1, 1, 1, 0, 0, 0],
                  [1, 0, 1, 1, 0, 0]])

    B = np.array([[1, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 1]])

    C = np.array([[1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 0],
                  [1, 1, 0]])

    # print(np.dot(A, B))

    m = sparse.csc_matrix(C)
    m[0, 0] = 0
    m.eliminate_zeros()
    print(m)


    # m = sparse.hstack([C, sparse.csr_matrix((4, 1), dtype='uint8')])
    # print(m.toarray())
    # print(*dir(m), sep='\n')


