import matplotlib.pyplot as plt


def plotErrorConvergence(outs=None, opts=None, *args, **kwargs):
    print('outs.residuals',outs.residuals)
     # Plot the error convergence curve
    if opts.recordReconErrors == True:
        fig = plt.figure()
        ax_1 = fig.add_subplot(111)
        ax_1.semilogy(outs.reconErrors)
        ax_1.set_xlabel('Iterations')
        ax_1.set_ylabel('ReconErrors')
        ax_1.set_title('Convergence curve:'+str(' ')+opts.algorithm)
        plt.show()

    if opts.recordResiduals == True:
        fig = plt.figure()
        ax_1 = fig.add_subplot(111)
        ax_1.semilogy(outs.residuals)
        ax_1.set_xlabel('Iterations')
        ax_1.set_ylabel('Residuals')
        ax_1.set_title('Convergence curve:'+str(' ')+opts.algorithm)
        plt.show()

    if opts.recordMeasurementErrors == True:
        fig = plt.figure()
        ax_1 = fig.add_subplot(111)
        ax_1.semilogy(outs.measurementErrors)
        ax_1.set_xlabel('Iterations')
        ax_1.set_ylabel('MeasurementErros')
        ax_1.set_title('Convergence curve:'+str(' ')+opts.algorithm)
        plt.show()

    return
