# -*- coding: utf-8 -*-
"""
Created on December 16, 2015
@author: pascal leimer
"""

import sys
from random import *
from math import *
import numpy as np
import numpy.random as rand
from optparse import OptionParser
from network import *
from utility import *

# ###### parameters #############
parser = OptionParser()
parser.add_option("--epsilon", dest="epsilon", type="float") # pattern similarity
parser.add_option("--epsilon2", dest="epsilon2", type="float", default=False) # roving: pattern similarity for second pattern class
parser.add_option("--alpha", dest="alpha", type="float") # associative task: context overlap; roving: spatial shift between easy and difficult patterns
parser.add_option("--a", dest="a", type="float") # midpoint of logistic function
parser.add_option("--beta", dest="beta", type="float") # steepness of logistic function
parser.add_option("--lambda1", dest="lambda1", type="float") # penalty factor on |V_i|
parser.add_option("--lambda2", dest="lambda2", type="float") # pentalty factor on w^f
parser.add_option("--eta", dest="eta", type="float") # learning rate
parser.add_option("--tau", dest="tau", type="float") # decay constant of low-pass filtered R and V
parser.add_option("--sigma", dest="sigma", type="float", default=0) # additive gaussian noise on inital w^f, w^s and input x
parser.add_option("--thres1", dest="thres1", type="float") # threshold on lowpass filtered potential
parser.add_option("--thres2", dest="thres2", type="float") # threshold on |w^f|
parser.add_option("--z", dest="z", type="float") # normalization factor of postsynaptic firing rates, use z=0 for sum_i(varphi(V_i))
parser.add_option("--t1", dest="t1", type="int") # learning time steps per context
parser.add_option("--t2", dest="t2", type="int") # break time steps
parser.add_option("--t3", dest="t3", type="int") # recall time steps
parser.add_option("--figures", dest="figures", type="int", default=1) # show figures at the end of simulation
parser.add_option("--runs", dest="runs", type="int", default=1) # number of simulation runs, results will be averaged
parser.add_option('--nI', dest="nI", type="int") # number of input neurons
parser.add_option('--nO', dest="nO", type="int") # number of output neurons
parser.add_option('--nO_active', dest="nO_active", type="int", default=False)
parser.add_option('--nP', dest="nP", type="int") # number of patterns
parser.add_option('--mode', dest="mode", type="string", default='"l"') # b=batch learning, l=online learning, r=roving, t=tagging experiments; for tagging append stimulus sequence
parser.add_option('--stimtype', dest="stimtype", type="string", default='"a"') # a=associative task stimuli, r=roving stimuli, t=tagging stimuli
(options, args) = parser.parse_args()
opts = vars(options)

para = ['epsilon', 'epsilon2', 'alpha', 'a', 'beta', 'lambda1', 'lambda2', 'eta', 'tau', 'sigma', 'thres1', 'thres2', 't1', 't2', 't3', 'z', 'figures', 'runs', 'nI', 'nO', 'nO_active', 'nP', 'mode', 'stimtype']
for p in para:
    if opts[p] != None:
        exec('%s = %s' % (p, opts[p]))


if len(mode) > 1:
    stimsequence = mode[1:]
    mode = mode[0]
else:
    stimsequence = []

nn = Network(nP, nI, nO, a, beta, lambda1, eta, lambda2, z, t1, t2, t3, tau, sigma, thres1, thres2, runs, mode, nO_active=nO_active)
nn.make_pattern(stimtype, epsilon, alpha, epsilon2)  # create input patterns

nn.set_stimsequence(stimsequence=stimsequence)  # define stimulus sequence
nn.run()  # run simulation

success = nn.calculateSuccess()  # brief evaluation of simulation
print(np.average(success))

if figures == 1:
    nn.plot_result()  # display plots