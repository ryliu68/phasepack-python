def displayVerboseOutput(iter=None, currentTime=None, currentResid=None, currentReconError=None, currentMeasurementError=None, *args, **kwargs):

    print('Iteration = %d', iter)
    print(' | IterationTime = %f', currentTime)
    if currentResid:
        print(' | Residual = %d', currentResid)

    if currentReconError:
        print(' | currentReconError = %d', currentReconError)

    if currentMeasurementError:
        print(' | MeasurementError = %d', currentMeasurementError)

    print('\n')

    return
