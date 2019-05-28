# -*- coding: utf-8 -*-
"""
Created on December 16, 2015
@author: pascal leimer
"""

from math import *
import numpy as np
import numpy.random as rand
import sys

vsign = np.vectorize(np.sign)


def vectorsignum(array):
    for i in range(len(array)):
        su = np.sum(array[i])
        array[i] = np.sign(su) if su != 0 else 0
    return array


def zStep(Z):
    if Z < 1.:
        return 1.
    else:
        return Z


def unitStep(x):
    if x == 0:
        return 0.
    else:
        return 0.5 * (np.sign(x) + 1)
vunitStep = np.vectorize(unitStep)


def flatten(array):
    return [item for sublist in array for item in sublist]


def sigmoid(x, a, beta):
    try:
        return 1. / (1. + exp(-(x - a) / beta))
    except:
        sys.exit('error sigmoid: ' + str(x) + ', '+ str(a) + ', ' + str(beta))
vsigmoid = np.vectorize(sigmoid)


def dsigmoid(x, a, beta):
    try:
        return (1. / beta) * exp(-(x - a) / beta) / (1 + exp(-(x - a) / beta))**2
    except:
        sys.exit('error dsigmoid: ' + str(x) + ', '+ str(a) + ', ' + str(beta))
vdsigmoid = np.vectorize(dsigmoid)


def lowpass(tau, old, new):
    if tau == 0:
        return new
    return old * exp(-1. / tau) + new * (1. - exp(-1. / tau))
vlowpass = np.vectorize(lowpass)


def addnoise(a, sigma):
    if sigma == 0:
        return a
    else:
        return a + rand.normal(0, sigma)
vaddnoise = np.vectorize(addnoise)