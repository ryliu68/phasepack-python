#                               getExtraOpts.m
#
# This function creates and outputs a struct that consists of all the
# options in opts but not in otherOpts.
# It is used as a helper function in manageOptions.m and
# manageOptionsForBenchmark.m
#
# Inputs:
#         opts(struct)       :  consists of options.
#         otherOpts(struct)  :  consists of options.
# Outputs:
#         extras(struct)     :  consists of extral options appearing in
#                               opts but not in otherOpts.
#
#
# PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
# Christoph Studer, & Tom Goldstein
# Copyright (c) University of Maryland, 2017

import struct


def getExtraOpts(opts=None, otherOpts=None, *args, **kwargs):
    extras = struct
    optNames = dir(opts)
    for i in range(1, len(optNames)):
        optName = optNames[i]
        print(hasattr(opts, optName))
        # if ~isfield(otherOpts, optName)
        if not hasattr(otherOpts, optName):
            setattr(extras, optName, getattr(opts, (optName)))
    return extras
