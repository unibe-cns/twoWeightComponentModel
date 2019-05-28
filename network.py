# -*- coding: utf-8 -*-
"""
Created on December 16, 2015
@author: pascal leimer
"""

import sys
from math import *
import numpy as np
import numpy.random as rand
import time
from utility import *
from plot import *

seed = int(time.time())
rand.seed(seed)


class Network:

    def __init__(self, nP, nI, nO, a, beta, lambda1, eta, lambda2, z, Tlearning, Tbreak, Trecall, tau, sigma, thres1, thres2, runs, mode='l', nO_active=False):
        self.nP = nP
        self.nI = nI
        self.nO = nO
        self.a = a
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eta = eta
        self.z = z
        self.Tlearning = Tlearning
        self.Tbreak = Tbreak
        self.Trecall = Trecall
        self.tau = tau
        self.sigma = sigma
        self.thres1 = thres1
        self.thres2 = thres2
        self.runs = runs
        self.success = np.zeros(self.nP)
        self.mode = mode

        if nO_active is False:  # no_active is used if for example input pattern 7 and 8 should activate output neuron 1 and 2. That means only 6 output neurons should be active
            self.nO_active = nO
        else:
            self.nO_active = nO_active

        if self.nP > self.nI:
            sys.exit('error: more patterns than input units')
        elif self.nP > self.nO:
            sys.exit('error: more patterns than output units')

    # defines input pattern
    def make_pattern(self, stimtype='a', epsilon=1, alpha=False, epsilon2=False):
        self.stimtype = stimtype
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon2 = epsilon2
        if epsilon2 is False:
            epsilon2 = epsilon

        if self.stimtype == 'r':
            self.x = np.zeros((self.nP, self.nI))
            alpha = int(alpha)
            for l in range(0, int(self.nI/2)):
                self.x[0, l] = 1 + epsilon
                self.x[1, l] = 1 - epsilon
            for l in range(int(self.nI/2), self.nI):
                self.x[0, l] = 1 - epsilon
                self.x[1, l] = 1 + epsilon
            if self.nP == 4:
                for l in range(alpha, int(self.nI/2+alpha)):
                    self.x[2, l] = 1 + epsilon2
                    self.x[3, l] = 1 - epsilon2
                for l in range(0, alpha):
                    self.x[2, l] = 1 - epsilon2
                    self.x[3, l] = 1 + epsilon2
                for l in range(int(self.nI/2+alpha), self.nI):
                    self.x[2, l] = 1 - epsilon2
                    self.x[3, l] = 1 + epsilon2

        elif self.stimtype == 't':
            self.x = np.zeros((4, self.nI))
            self.x[0, 0] = 1.
            self.x[1, 1] = 1.
            self.x[2, 0] = 1.
            self.x[3, 1] = 1.

        elif self.stimtype == 'a':
            self.x = np.ones((self.nP, self.nI)) * alpha
            for i in range(0, self.nP):
                self.x[i][i] = 1
                self.x[i][(i + 1) % 2 + 2 * int(i / 2)] = 1 - epsilon

        else:
            sys.exit('error: no valid stimtype defined')

    # defines the sequence in which the stimuli are applied and writes it to self.sessiontime. 
    # if multiple simulation runs are called at once, set_stimsequence() is called before each run
    def set_stimsequence(self, stimsequence=[], Tplastic=None):
        if len(stimsequence) and self.mode != 't':
            self.sessiontime = stimsequence
            self.T = len(stimsequence)
            self.Tplastic = Tplastic
        else:
            if self.mode in ['l', 'r', 'b']:  # l=online learning, r=roving, b=batch
                nC = int(np.ceil(self.nP / 2.))  # number of contexts
                self.T = nC * (self.Tlearning + self.Tbreak) + self.Trecall  # total simulation steps
                if self.mode == 'r':
                    self.T -= self.Tbreak  # correct total simulation steps for roving (no break)
                self.Tplastic = self.T - self.Trecall  # total steps where synapse are plastic
                self.sessiontime = np.zeros(self.T)  # stimulus order
                t = 0
                for c in range(1, nC+1):
                    for j in range(self.Tlearning):
                        if self.mode == 'r':
                            self.sessiontime[t] = rand.randint(self.nO_active) + 1
                        else:
                            self.sessiontime[t] = int(rand.choice([2*c-1, 2*c]))
                        t += 1
                    if (self.mode == 'r' and c == 2) or (self.mode != 'r'):  # remove break after 1st context in roving
                        for j in range(self.Tbreak):
                            t += 1
                for c in range(1, nC+1):
                    for j in range(int(self.Trecall/nC)):
                        self.sessiontime[t] = (-1)*int(rand.choice([2*c-1, 2*c]))
                        t += 1

            elif self.mode == 't':  # tagging
                if stimsequence == 'a':  # weak LTP
                    self.sessiontime = np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                elif stimsequence == 'b':  # strong LTP
                    self.sessiontime = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                elif stimsequence == 'c':  # weak LTD
                    self.sessiontime = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                elif stimsequence == 'd':  # strong LTD
                    self.sessiontime = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                elif stimsequence == 'ab':  # weak LTP - strong LTP
                    self.sessiontime = np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                elif stimsequence == 'ba':  # strong LTP - weak LTP
                    self.sessiontime = np.array([0,0,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                elif stimsequence == 'cd':  # weak LTD - strong LTD
                    self.sessiontime = np.array([0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                elif stimsequence == 'dc':  # strong LTD - weak LTD
                    self.sessiontime = np.array([0,0,0,0,0,0,0,0,0,0,0,4,4,4,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                elif stimsequence == 'bc':  # strong LTP - weak LTD
                    self.sessiontime = np.array([0,0,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                elif stimsequence == 'cb':  # weak LTD - strong LTP
                    self.sessiontime = np.array([0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                elif stimsequence == 'ad':  # weak LTP - strong LTD
                    self.sessiontime = np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                elif stimsequence == 'da':  # strong LTD - weak LTP
                    self.sessiontime = np.array([0,0,0,0,0,0,0,0,0,0,0,4,4,4,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                else:
                    sys.exit('no protocol defined')
                self.T = self.Tplastic = len(self.sessiontime)  # no recall and breaks during tagging experiments

    # one batch learning step
    def _learnStepBatch(self, pt_t, pause, age):
        # pause = 0: learning
        # pause = 1: break
        # pause = -1: recall

        if pause == 0 or pause == -1:  # no pause, i.e. pattern presentation
            delta = np.zeros((self.nO, self.nI))
            reward = 0
            for pt in range(self.nP):
                inp = self.x[int(pt)]
                v = np.dot(self.wF + self.wS, np.transpose(inp))  # membrane potential
                r = vsigmoid(v, self.a, self.beta)  # firing rate
                dr = vdsigmoid(v, self.a, self.beta)  # derivative of firing rate with respect to w
                props = r / np.sum(r)
                zz = np.sum(r)
                for winner in range(self.nO):
                    xi = np.full(self.nO, -r[winner] / zz)
                    xi[winner] = xi[winner] + 1
                    tempreward = 1 if winner == pt else 0
                    reward += 1./self.nP * props[winner] * tempreward
                    delta += 1./self.nP * props[winner] * tempreward * np.outer(dr / r[winner] * xi, inp) - self.lambda1 * np.outer(vsign(v), inp) - self.lambda2 * self.wF
            inp_t = self.x[int(pt_t)]
            v_t = np.dot(self.wF + self.wS, np.transpose(inp_t))  # membrane potential
            r_t = vsigmoid(v_t, self.a, self.beta)  # firing rate
            zz_t = np.sum(r_t)
            props_t = np.asarray(r_t / zz_t)
            winner_t = rand.choice(np.arange(self.nO), p=props_t)  # get one winner: only for statistics
            reward_t = 1 if winner_t == pt_t else 0
            xi_t = np.full(self.nO, -r_t / zz_t)
            xi_t[winner_t] = xi_t[winner_t] + 1
        else:
            v_t = np.zeros(self.nO)
            r_t = vsigmoid(v_t, self.a, self.beta)  # firing rate
            zz_t = np.sum(r_t)
            reward_t = 0
            reward = 0
            xi_t = [None]
            delta_t = np.zeros((self.nO, self.nI))
            delta = np.zeros((self.nO, self.nI))
            winner_t = -1

        timeconstant = np.amin([self.tau, age])
        self.rewardMean = lowpass(timeconstant, self.rewardMean, reward)

        if pause != -1:
            self.vMean = vlowpass(self.tau, self.vMean, v_t)  # low-pass filtered potential
            wFnew = self.wF + self.eta * delta
            self.wS += (self.lambda2 * self.wF) * (vunitStep(self.vMean - self.thres1)).reshape(self.nO, 1) * vunitStep(np.absolute(self.wF) - self.thres2)
            self.wF = wFnew
        return v_t, r_t, reward_t, winner_t, zz_t, xi_t

    # one online learning step
    def _learnStepOnline(self, pt, pause, age):
        # pause = 0: learning
        # pause = 1: break
        # pause = -1: recall

        if pt == -1:  # no pattern is presented
            inp = np.zeros(self.nI)
        else:
            inp = vaddnoise(self.x[int(pt)], self.sigma)
        if pause == 0 or pause == -1:  # no pause, i.e. pattern presentation
            v = np.dot(self.wF + self.wS, np.transpose(inp))  # membrane potential
            r = vsigmoid(v, self.a, self.beta)  # firing rate
            dr = vdsigmoid(v, self.a, self.beta)  # derivative of firing rate with respect to w
            props = np.asarray(r / np.sum(r))  # probability to win
            winner = rand.choice(np.arange(self.nO), p=props)  # get one winner
            # winner = np.argmax(v)
            if self.z == 0:
                zz = np.sum(r)
            else:
                zz = self.z
            xi = -r / zz
            if self.mode != 't' or pt < 2:  # tagging: pt=0 or 1 is LTP, pt=2 or 3 is LTD
                xi[winner] = xi[winner] + 1
            if self.nO == 1:
                reward = 1
            else:
                reward = 1 if winner == (pt % self.nO_active) else 0
            delta = (reward - self.rewardMean) * np.outer(dr / r * xi, inp)

        else:
            v = np.zeros(self.nO)
            r = vsigmoid(v, self.a, self.beta)  # firing rate
            reward = 0
            xi = [None]
            delta = np.zeros((self.nO, self.nI))
            winner = -1
        if self.mode == 't':
            self.rewardMean = 0.
        else:
            timeconstant = np.amin([self.tau, age])
            self.rewardMean = lowpass(timeconstant, self.rewardMean, reward)  # low-pass filtered reward
        if pause != -1:
            self.vMean = vlowpass(self.tau, self.vMean, v)  # low-pass filtered potential
            self.vFMean = vlowpass(self.tau, self.vFMean, np.dot(self.wF, np.transpose(inp)))
            self.vSMean = vlowpass(self.tau, self.vSMean, np.dot(self.wS, np.transpose(inp)))
            wFnew = self.eta * delta - self.lambda1 * np.outer(vsign(v), inp) - self.lambda2 * self.wF
            self.wS += (self.lambda2 * self.wF) * (vunitStep(self.vMean - self.thres1)).reshape(self.nO, 1) * vunitStep(np.absolute(self.wF) - self.thres2)
            self.wF = self.wF + wFnew
        return v, r, reward, winner, np.sum(r), xi

    def _learnStep(self, pt, age, pause=0):
        if self.mode == 'b':
            return self._learnStepBatch(pt, pause, age)
        else:
            return self._learnStepOnline(pt, pause, age)

    def run(self, initialWeights=None):
        self.totalRim = np.zeros(self.T)
        self.totalR = np.zeros(self.T)
        self.totalZ = np.zeros(self.T)
        self.totalWf = [[np.zeros(self.Tplastic) for _ in range(self.nI)] for _ in range(self.nO)]
        self.totalWs = [[np.zeros(self.Tplastic) for _ in range(self.nI)] for _ in range(self.nO)]
        self.totalVmean = [np.zeros(self.Tplastic) for _ in range(self.nO)]
        self.totalVFmean = [np.zeros(self.Tplastic) for _ in range(self.nO)]
        self.totalVSmean = [np.zeros(self.Tplastic) for _ in range(self.nO)]
        self.totalV = [np.zeros(self.Tplastic) for _ in range(self.nO)]
        self.totalRate = [np.zeros(self.Tplastic) for _ in range(self.nO)]
        self.patternRecord = []
        # run it over multiple trials
        for _ in range(self.runs):
            if self.runs > 1 and self.mode != 't':
                self.set_stimsequence()
            self.humps = np.zeros((self.nP, self.nO))
            age = 0
            nRecall = np.zeros(self.nP)
            # initialize weights
            self.wF = vaddnoise(np.zeros((self.nO, self.nI)), self.sigma)
            if initialWeights is not None:
                self.wS = initialWeights
            else:
                self.wS = vaddnoise(np.zeros((self.nO, self.nI)), self.sigma)
            self.vMean = np.zeros(self.nO)
            self.vFMean = np.zeros(self.nO)
            self.vSMean = np.zeros(self.nO)
            self.tBlock = np.zeros(self.nO)
            if self.mode == 't':
                self.rewardMean = 0
            else:
                self.rewardMean = 1. / self.nO

            # do T learning steps
            for t in np.arange(0, self.T):
                if self.sessiontime[t] == 0:  # break
                    age = 0
                    v, r, reward, winner, z, xi = self._learnStep(-1, t, pause=1)
                elif self.sessiontime[t] > 0:  # learn
                    pt = self.sessiontime[t] - 1
                    v, r, reward, winner, z, xi = self._learnStep(pt, t)
                    self.patternRecord.append([winner, t, reward])
                    age += 1
                elif self.sessiontime[t] < 0:  # recall
                    pt = int((-1) * self.sessiontime[t] - 1)
                    foo1, foo2, reward, winner, z, foo3 = self._learnStep(pt, t, pause=-1)
                    self.humps[pt, winner] += reward
                    nRecall[pt] += 1
                    age += 1
                else:
                    sys.exit(0)
                self.totalRim[t] += reward / float(self.runs)
                self.totalR[t] += self.rewardMean / float(self.runs)
                self.totalZ[t] += z / float(self.runs)
                if t < self.Tplastic:
                    for i in range(self.nO):
                        for j in range(self.nI):
                            self.totalWf[i][j][t] += self.wF[i][j] / float(self.runs)
                            self.totalWs[i][j][t] += self.wS[i][j] / float(self.runs)
                        self.totalVmean[i][t] += self.vMean[i] / float(self.runs)
                        self.totalVFmean[i][t] += self.vFMean[i] / float(self.runs)
                        self.totalVSmean[i][t] += self.vSMean[i] / float(self.runs)
                        self.totalV[i][t] += v[i] / float(self.runs)
                        self.totalRate[i][t] += r[i] / float(self.runs)

            if self.mode != 't':
                for pt in range(0, self.nP):
                    self.success[pt] += float(100 / nRecall[pt] * self.humps[pt, pt % self.nO_active]) / float(self.runs)

    def calculateSuccess(self):
        success = [np.mean(self.success)]
        for pt in range(self.nP):
            success.append(self.success[pt])
        return success

    def show_result(self):
        print('--------')
        success = self.calculateSuccess()
        print('success rates total: %3.1f %%' % success[0])
        for pt in range(1, self.nP+1):
            print ('pattern ' + str(pt) + ': %3.1f %%' % success[pt])
        print('--------')

    def show_finalWeights(self):
        for i in range(self.nO):
            for j in range(self.nI):
                print(self.totalWf[i][j][-1], self.totalWs[i][j][-1], self.totalWf[i][j][-1]+self.totalWs[i][j][-1])
            print('-')

    def print_result(self, filename):
        f1 = open(filename, 'a')
        success = self.calculateSuccess()
        if self.epsilon2 is False:
            f1.write('{:04.2f}\t{:04.2f}\t{:04.1f}'.format(self.alpha, self.epsilon, success[0]))
        else:
            f1.write('{:04.2f}\t{:04.2f}\t{:04.2f}\t{:04.1f}'.format(self.alpha, self.epsilon, self.epsilon2, success[0]))
        for pp in range(self.nP/2):
            f1.write('\t{:04.1f}'.format(success[pp*2+1]/2.+success[pp*2+2]/2.))
        f1.write('\n')
        f1.close()

    def print_rewardtrace(self, filename):
        f1 = open(filename, 'w')
        for t in range(self.T):
            f1.write(str(self.totalR[t]) + '\n')

    def print_weighttrace(self, filename):
        f1 = open(filename, 'w')
        for t in range(self.Tplastic):
            line = ''
            for i in range(self.nO):
                for j in range(self.nI):
                    line += str(self.totalWf[i][j][t])+'\t'+str(self.totalWs[i][j][t])+'\t'
            line += '\n'
            f1.write(line)

    def print_finalWeights(self, filename):
        f1 = open(filename, 'w')
        text = ''
        for i in range(self.nO):
            for j in range(self.nI):
                text += str(self.totalWf[i][j][-1]+self.totalWs[i][j][-1])+','
            text = text[:-1]
            text += '\n'
        f1.write(text)

    def plot_result(self):
        pl = Plot(self.T, self.Tplastic, self.x, self.nI, self.nO, self.nP, self.a, self.beta, self.sessiontime, self.totalWf, self.totalWs, self.totalR, self.totalV, self.totalZ, self.patternRecord, self.humps, self.thres1, self.thres2, self.totalVmean)
        if self.mode == 't':
            pl.taggingPlots()
        elif self.nP == 1 or self.nP == 2:
            pl.twoPatternPlots()
        else:
            pl.fourPatternPlots()
        plt.show()
