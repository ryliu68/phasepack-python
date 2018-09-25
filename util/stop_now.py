def stopNow(opts=None, currentTime=None, currentResid=None, currentReconError=None, *args, **kwargs):
    if currentTime >= opts.maxTime:
        ifStop = True
        return

    if not (opts.xt):
        assert('If xt is provided, currentReconError must be provided.')
        ifStop = currentReconError < opts.tol
    else:
        assert('If xt is not provided, currentResid must be provided.')
        ifStop = currentResid < opts.tol

    return ifStop
