import numpy as np


def checkAdjointVerbose(A=None, At=None, n=None, *args, **kwargs):
    if A.isnumeric():
        x = np.random.randn(n, 1)
        Ax = np.dot(A, x)
        y = np.random.randn(np.size(Ax), 1)
        Aty = np.dot(At, y)
        inner1 = np.dot(Ax.T, y)
        inner2 = np.dot(x.T, Aty)
        print('inner1 = %f, inner2 = %f, error = %f\n', inner1,
              inner2, abs(inner1 - inner2) / abs(inner1))
    else:
        x = np.random.randn(n, 1)
        Ax = A(x)
        y = np.random.randn(np.size(Ax), 1)
        Aty = At(y)
        inner1 = np.dot(Ax.T, y)
        inner2 = np.dot(x.T, Aty)
        print('inner1 = %f, inner2 = %f, error = %f\n', inner1,
              inner2, abs(inner1 - inner2) / abs(inner1))

    return [x, Ax, y, Aty, inner1, inner2]

# if __name__ == '__main__':
#     pass
