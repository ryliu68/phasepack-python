import matplotlib.pyplot as plt
import numpy as np


def plotRecoveredVSOriginal(x=None, xt=None,):
    fig = plt.figure()
    ax_1 = fig.add_subplot(111)
    ax_1.scatter(np.real(x), np.real(xt), c='b', marker='o')
    plt.plot([-3, 3], [-3, 3])
    ax_1.set_title('Visual Correlation of Recovered signal with True Signal')
    ax_1.set_xlabel('Recovered Signal')
    ax_1.set_ylabel('True Signal')
    plt.show()
    return
