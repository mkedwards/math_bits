# math_bits

#!python
import numpy as np
from numpy.linalg import svd


def rank(A, atol=1e-13, rtol=0):
    """Estimate the rank (i.e. the dimension of the nullspace) of a matrix.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length n will be treated
        as a 2-D with shape (1, n)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    r : int
        The estimated rank of the matrix.

    See also
    --------
    numpy.linalg.matrix_rank
        matrix_rank is basically the same as this function, but it does not
        provide the option of the absolute tolerance.
    """

    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    return rank


def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


#!python
import numpy as np

from rank_nullspace import rank, nullspace

np.set_printoptions(linewidth=10000,suppress=True)

def checkit(a):
    print "a:"
    print a
    r = rank(a)
    print "rank is", r
    ns = nullspace(a)
    ns = np.around(3 * ns / ns[0], decimals=6)
    print "nullspace:"
    print ns
    if ns.size > 0:
        res = np.abs(np.dot(a, ns)).max()
        print "max residual is", res
    print "eigenvalues and eigenvectors:"
    res = np.linalg.eigh(a)
    scale = np.stack((-res[1].min(0), res[1].max(0))).max(0)
    print np.around(res[0], decimals=6)
    print np.around(res[1] / scale, decimals=6)
    psinv = np.linalg.pinv(a, rcond=1e-14)
    adj = np.matmul(ns, np.transpose(ns))
    fudge = adj * (-227. / 2880.)
    psi = np.add(psinv, fudge)
    print "adj (= ns * ns^T):"
    print adj
    adj2 = np.matmul(adj, adj)
    print "scale of adj is", adj2[0][0] / adj[0][0]
    d = np.linalg.det(psi)
    print "determinant of psi is", 2880. * d, " / 2880"
    print "-240 * psi:"
    print np.around(-240 * psi, decimals=6)
    print "eigenvalues and eigenvectors:"
    res = np.linalg.eigh(psi)
    scale = np.stack((-res[1].min(0), res[1].max(0))).max(0)
    print np.around(res[0], decimals=6)
    print np.around(res[1] / scale, decimals=6)
    print "-(psi + flipud(psi)):"
    print np.around(-np.add(psi, np.flipud(psi)), decimals=6)
    proj240 = np.around(240 * np.matmul(psi, a), decimals=6)
    proj = proj240 * (1./240.)
    print "projection * 240:"
    print proj240
    print "projection^2 - projection:"
    print np.around(np.add(np.matmul(proj, proj), -proj), decimals=6)
    print "trace of projection is", np.trace(proj)
    scaled = np.matmul(psi, ns)
    scalar = (scaled[0] / ns[0])[0]
    print "(psi - ", scalar, ") * null vector of a:"
    print np.add(scaled, (-scalar) * ns)
    print "a * psi - psi * a:"
    print np.around(np.add(np.matmul(a, psi), -np.matmul(psi, a)), decimals=6)
    print "(a * psi) + adj/240:"
    print np.around(np.add(np.matmul(a, psi), adj * (1. / 240.)), decimals=6)
    #print np.around(240 * (np.identity(psi.shape[0]) - np.matmul(psi, a)), decimals=6)
    print "adj * a:"
    print np.around(np.matmul(adj, a), decimals=6)
    print "a * adj:"
    print np.around(np.matmul(a, adj), decimals=6)
    print "a * psi * a - a:"
    print np.around(np.add(np.matmul(np.matmul(a, psi), a), -a), decimals=6)
    print "(psi * a * psi) - (psi + 227/2880 * adj):"
    print np.around(np.add(np.matmul(np.matmul(psi, a), psi), -np.add(psi, -fudge)), decimals=6)
    psinvinv = np.linalg.pinv(psi, rcond=1e-14)
    print "(psi^-1 - a) * 227 * 20:"
    print np.around((227. * 20.) * np.add(psinvinv, -a), decimals=6)
    print "psi on ones:"
    print np.around(np.matmul(psi, np.ones((psi.shape[0], 1))), decimals=6)
    print "a on psi on ones:"
    print np.around(np.matmul(a, np.matmul(psi, np.ones((psi.shape[0], 1)))), decimals=6)


print "-"*25

#a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
#checkit(a)

#print "-"*25

#a = np.array([[0.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
#checkit(a)

#print "-"*25

#a = np.array([[0.0, 1.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
#checkit(a)

#print "-"*25

#a = np.array([[1.0,   1.0j,   2.0+2.0j],
#              [1.0j, -1.0,   -2.0+2.0j],
#              [0.5,   0.5j,   1.0+1.0j]])
#checkit(a)

#print "-"*25

a = np.array(
             [
              [2, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [-1, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, -1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 2]
             ]
            )
checkit(a)

print "-"*25

#a = np.array(
#             [
#              [2, 0, 0, -1, 0, 0, 0, 0, 0, 0],
#              [0, 2, -1, 0, 0, 0, 0, 0, 0, 0],
#              [0, -1, 2, -1, 0, 0, 0, 0, 0, 0],
#              [-1, 0, -1, 2, -1, 0, 0, 0, 0, 0],
#              [0, 0, 0, -1, 2, -1, 0, 0, 0, 0],
#              [0, 0, 0, 0, -1, 2, -1, 0, 0, 0],
#              [0, 0, 0, 0, 0, -1, 2, -1, 0, 0],
#              [0, 0, 0, 0, 0, 0, -1, 2, -1, 0],
#              [0, 0, 0, 0, 0, 0, 0, -1, 2, -1],
#              [0, 0, 0, 0, 0, 0, 0, 0, -1, 2]
#             ]
#            )
#checkit(a)

#print "-"*25

#a = np.array(
#             [
#              [2, 0, 0, -1, 0, 0, 0, 0, 0, 0],
#              [0, 2, -1, 0, 0, 0, 0, 0, 0, 0],
#              [0, -1, 2, -1, 0, 0, 0, 0, 0, 0],
#              [-1, 0, -1, 2, -1, 0, 0, 0, 0, 0],
#              [0, 0, 0, -1, 2, -1, 0, 0, 0, 0],
#              [0, 0, 0, 0, -1, 2, -1, 0, 0, 0],
#              [0, 0, 0, 0, 0, -1, 2, -1, 0, 0],
#              [0, 0, 0, 0, 0, 0, -1, 2, -1, 0],
#              [0, 0, 0, 0, 0, 0, 0, -1, 2, -1],
#              [0, 0, 0, 0, 0, 0, 0, 0, -2, 2]
#             ]
#            )
#checkit(a)

#print "-"*25
#
#a = np.array(
#             [
# [  225,   235,   464,   681,   655,   614,   561,   499,   431,   360,   289,   221,   159,   106,    65,    39,    16,     5,    15],
# [  235,     0,   236,   464,   445,   416,   379,   336,   289,   240,   191,   144,   101,    64,    35,    16,     4,     0,     5],
# [  464,   236,   464,   916,   880,   824,   752,   668,   576,   480,   384,   292,   208,   136,    80,    44,    16,     4,    16],
# [  681,   464,   916,  1344,  1295,  1216,  1113,   992,   859,   720,   581,   448,   327,   224,   145,    96,    44,    16,    39],
# [  655,   445,   880,  1295,  1025,   970,   895,   805,   705,   600,   495,   395,   305,   230,   175,   145,    80,    35,    65],
# [  614,   416,   824,  1216,   970,   704,   662,   608,   546,   480,   414,   352,   298,   256,   230,   224,   136,    64,   106],
# [  561,   379,   752,  1113,   895,   662,   417,   403,   383,   360,   337,   317,   303,   298,   305,   327,   208,   101,   159],
# [  499,   336,   668,   992,   805,   608,   403,   192,   217,   240,   263,   288,   317,   352,   395,   448,   292,   144,   221],
# [  431,   289,   576,   859,   705,   546,   383,   217,    49,   120,   191,   263,   337,   414,   495,   581,   384,   191,   289],
# [  360,   240,   480,   720,   600,   480,   360,   240,   120,     0,   120,   240,   360,   480,   600,   720,   480,   240,   360],
# [  289,   191,   384,   581,   495,   414,   337,   263,   191,   120,    49,   217,   383,   546,   705,   859,   576,   289,   431],
# [  221,   144,   292,   448,   395,   352,   317,   288,   263,   240,   217,   192,   403,   608,   805,   992,   668,   336,   499],
# [  159,   101,   208,   327,   305,   298,   303,   317,   337,   360,   383,   403,   417,   662,   895,  1113,   752,   379,   561],
# [  106,    64,   136,   224,   230,   256,   298,   352,   414,   480,   546,   608,   662,   704,   970,  1216,   824,   416,   614],
# [   65,    35,    80,   145,   175,   230,   305,   395,   495,   600,   705,   805,   895,   970,  1025,  1295,   880,   445,   655],
# [   39,    16,    44,    96,   145,   224,   327,   448,   581,   720,   859,   992,  1113,  1216,  1295,  1344,   916,   464,   681],
# [   16,     4,    16,    44,    80,   136,   208,   292,   384,   480,   576,   668,   752,   824,   880,   916,   464,   236,   464],
# [    5,     0,     4,    16,    35,    64,   101,   144,   191,   240,   289,   336,   379,   416,   445,   464,   236,     0,   235],
# [   15,     5,    16,    39,    65,   106,   159,   221,   289,   360,   431,   499,   561,   614,   655,   681,   464,   235,   225]
#             ]
#            )
#checkit(a)
