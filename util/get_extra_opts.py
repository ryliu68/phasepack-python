import struct


def getExtraOpts(opts=None, otherOpts=None):
    extras = struct
    optNames = dir(opts)
    for i in range(1, len(optNames)):
        optName = optNames[i]
        if not hasattr(otherOpts, optName):
            setattr(extras, optName, getattr(opts, (optName)))
    return extras

