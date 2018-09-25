def applyOpts(opts=None, otherOpts=None, override=None, *args, **kwargs):
    otherOptNames = dir(otherOpts)
    for i in range(1, len(otherOptNames)):
        optName = otherOptNames[i]
        if not (hasattr(opts, optName)) or override:
            setattr(opts, optName, getattr(otherOpts, (optName)))

    return opts
 