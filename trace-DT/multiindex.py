import numpy as np


def multiindex(m, p, full_tensor=False):
    d = np.array(range(p + 1))
    M = np.meshgrid(*[d] * m)
    I = np.concatenate([i.flatten().reshape(-1, 1) for i in M], axis=1)

    if not full_tensor:
        I = I[np.sum(I, axis=1) <= p, :]
    I = np_sortrows(I, list(range(m - 1, 0, -1)))
    ind = np.argsort(np.sum(I, axis=1))
    I = I[ind, :]
    return I


class Descending:
    """ for np_sortrows: sort column in descending order """

    def __init__(self, column_index):
        self.column_index = column_index

    def __int__(self):  # when cast to integer
        return self.column_index


def np_sortrows(M, columns=None):
    """  sorting 2D matrix by rows
    :param M: 2D numpy array to be sorted by rows
    :param columns: None for all columns to be used,
                    iterable of indexes or Descending objects
    :return: returns sorted M
    """
    if len(M.shape) != 2:
        raise ValueError('M must be 2d numpy.array')
    if (columns is None) or len(columns)==0:  # no columns specified, use all in reversed order
        M_columns = tuple(M[:, c] for c in range(M.shape[1] - 1, -1, -1))
    else:
        M_columns = []
        for c in columns:
            M_c = M[:, int(c)]
            if isinstance(c, Descending):
                M_columns.append(M_c[::-1])
            else:
                M_columns.append(M_c)
        M_columns.reverse()

    return M[np.lexsort(M_columns), :]

    # if full_tensor:
    #     I = multiindex_full(m, p)
    # else:
    #     I = multiindex_complete(m, p)


# def multiindex_full(m, p):
#     if p == 0 or p == 1:
#         I = np.zeros([1,m])
#         if p == 1:
#             I = np.concatenate(I, np.eye(m))
#     else:
#         I_kp = []
#         for q in range(p+1):
#             I_kp.append(np.zeros([m,0]))
#
#     return I

# def multiindex_complete(m, p):
#     if p == 0:
#         I_kp = [np.zeros(1, m)]
#         return I_kp
#     elif p == 1:
#         I_kp = [np.zeros(1,m),  np.eye(m,m)]
#         return I_kp
#
#     I_kp = [None]*(p+1)
#     for q in range(p+1):
#         I_kp[q] = np.zeros([q==0, 0])
#     # Iterate over the number of random variables
#     for k in range(1,m+1):
#         # Backup the old multiindex set for later use
#         I_k1p = I_kp
#     return I_kp
#
# def multiindex_stats(m, p):
#     count = np.ones([p+1])
#     nonzero = np.concatenate([np.zeros([1]), np.ones([p])], axis = 0)
#     for k in range(2, m+1):
#         for q in range(p, -1, -1):
#             count[q] = np.sum(count[0:q+1])
#             nonzero[q] = np.sum(nonzero[0:q+1]) + np.sum(count[0:q])
#     return count, nonzero
#

def main():
    print(multiindex(3, 4))


if __name__ == "__main__":
    main()
