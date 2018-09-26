import warnings


def warnExtraOpts(extras):
    optNames = dir(extras)
    for i in range(len(optNames)):
        optName = optNames[i]
        warnings.warn('Provided option "{0}" is invalid and will be ignored'.format(optName))
