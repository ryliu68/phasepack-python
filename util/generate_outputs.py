import struct


def generateOutputs(opts=None, itera=None, solveTimes=None, measurementErrors=None, reconErrors=None, residuals=None, *args, **kwargs):
    outs = struct
    if type(solveTimes) != None:
        outs.solveTimes = solveTimes[1:itera]

    if measurementErrors:
        outs.measurementErrors = measurementErrors[1:itera]

    if reconErrors:
        outs.reconErrors = reconErrors[1:itera]

    if type(residuals) != None:
        outs.residuals = residuals[1:itera]

    outs.iterationCount = itera
    return outs