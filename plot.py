# -*- coding: utf-8 -*-
"""
Created on September 02, 2016
@author: pascal leimer
"""

from math import *
import numpy as np
import numpy.random as rand
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
#import pickle as pl
from utility import *


#colors from http://spartanideas.msu.edu/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


class Plot:
    def __init__(self, T, Tplastic, x, nI, nO, nP, a, beta, sessiontime, wF, wS, totalR, totalV, z, patternRecord, humps, thres1, thres2, vMean):
        self.T = T
        self.Tplastic = Tplastic
        self.x = x
        self.nI = nI
        self.nO = nO
        self.nP = nP
        self.a = a
        self.beta = beta
        self.sessiontime = sessiontime
        self.wF = wF
        self.wS = wS
        self.totalR = totalR
        self.totalV = totalV
        self.z = z
        self.patternRecord = patternRecord
        self.humps = humps
        self.thres1 = thres1
        self.thres2 = thres2
        self.vMean = vMean

        self.trange = np.arange(self.Tplastic)
        self.trange2 = np.arange(self.T)
        self.sessiontime_cut = self.sessiontime[0:self.Tplastic]

        # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
        for i in range(len(tableau20)):
            re, g, b = tableau20[i]
            tableau20[i] = (re / 255., g / 255., b / 255.)

    def _improveStyle(self, ax):
        ax.spines['bottom'].set_color('#5F5F5F')
        ax.spines['left'].set_color('#5F5F5F')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.xaxis.label.set_color('#5F5F5F')
        ax.yaxis.label.set_color('#5F5F5F')
        ax.tick_params(colors='#5F5F5F')
        ax.title.set_color('#5F5F5F')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 3))
        lg = ax.legend(loc=0, prop={'size': 11}, labelspacing=0.2)
        lg.draw_frame(False)

    def _sumsort(self, data, data2=False, data3=False, data4=False):
        sum = [(-1)*np.sum(data[:, i]) for i in range(len(data[0]))]
        data = np.transpose([data[:, i] for i in np.argsort(sum)])
        if not data2:
            return data
        data2 = [data2[i] for i in np.argsort(sum)]
        if not data3:
            return data, data2
        data3 = [data3[i] for i in np.argsort(sum)]
        if not data4:
            return data, data2, data3
        data4 = [data4[i] for i in np.argsort(sum)]
        return data, data2, data3, data4

    def _humps(self, ax):
        n = len(self.humps[0])

        # necessary variables
        ind = np.arange(n)
        width = 0.15

        bars = []
        # the bars
        for i in range(len(self.humps)):
            foo = ax.bar(ind+i*width, self.humps[i], width, color=tableau20[i])
            bars.append(foo[0])

        # axes and labels
        ax.set_xlim(-width, len(ind)+width)
        ax.set_ylabel('')
        ax.set_title('')
        xTickMarks = ['U'+str(i+1) for i in range(n)]
        ax.set_xticks(ind+width)
        xtickNames = ax.set_xticklabels(xTickMarks)
        plt.setp(xtickNames, rotation=0, fontsize=10)

        # add a legend
        ax.legend(list(bars), ['pt '+str(i+1) for i in range(len(self.humps))], loc=0, prop={'size': 11}, labelspacing=0.2)

    def _weightTrajectory(self, ax, data, labels):
        for i in range(len(data)):
            ax.plot(self.trange, data[i], color=tableau20[i], label=labels[i])
        ax.set_xlim(0, self.Tplastic)
        ax.set_title('')
        ax.set_xlabel('t')
        ax.set_ylabel('')
        ax.legend(loc=0)

    def _weightSpace(self, ax, data, labels, title, xlabel, ylabel):
        diagonal = np.arange(-15, 15, 1)
        diagonal2 = np.arange(15, -15, -1)
        for i in range(len(data)):
            ax.plot(data[i][0], data[i][1], color=tableau20[i], label=labels[i])
        ax.plot(diagonal, diagonal2, color="#000000")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc=0)

    def _activationFunction(self, ax):
        #plot distribution of potential together with the activation function
        self.totalV = list(filter(lambda a: a != 0., flatten(self.totalV)))
        min = np.min(self.totalV)
        max = np.max(self.totalV)
        potSpec = []
        for i in range(100):
            potSpec.append(0)
        for i in range(len(self.totalV)):
            try:
                bin = int((self.totalV[i] - min) / (max - min) * 99)
                potSpec[bin] += 1
            except:
                pass

        ax.bar(np.arange(100), potSpec, 1, color=tableau20[14], label=' ')
        ax.set_xlabel('V')

    def _membranePotential(self, ax, pt, neurons):
        #plot membrane potential for a given input pattern.
        #Note: The graphs are continues despite presenting patterns alternately
        pot = [0 for _ in range(self.nO)]
        if pt == 0 or pt == 1:
            showtime = [(1 if s == 1 or s == -1 else 0) for s in self.sessiontime]
        elif pt == 2 or pt == 3:
            showtime = [(1 if s == 2 or s == -1 else 0) for s in self.sessiontime]
        for i in range(self.nO):
            for j in range(self.nI):
                pot[i] += (self.wF[i][j] + self.wS[i][j]) * self.x[pt, j] * 1/self.nI  # * showtime

        for i in range(len(neurons)):
            ax.plot(self.trange, pot[neurons[i]], color=tableau20[i], label='neuron '+str(neurons[i]+1))
        ax.set_xlim(0, self.Tplastic)
        ax.set_title('membrane potential')
        ax.set_ylabel('pattern '+str(pt+1))

    def _diffmembranePotential(self, ax, neurons):
        #plot difference of membrane potentials for the first tow input patterns
        #Note: The graphs are continues despite presenting patterns alternately

        pot1 = 0
        pot2 = 0
        showtime = [(1 if s == 1 or s == -1 else 0) for s in self.sessiontime]
        for j in range(self.nI):
            pot1 += (self.wF[0][j] + self.wS[0][j]) * self.x[0, j] * 1./self.nI - (self.wF[1][j] + self.wS[1][j]) * self.x[0, j] * 1./self.nI
            pot2 += (self.wF[0][j] + self.wS[0][j]) * self.x[1, j] * 1./self.nI - (self.wF[1][j] + self.wS[1][j]) * self.x[1, j] * 1./self.nI
        pot = np.abs(vunitStep(pot1)-vunitStep(pot2))
        ax.plot(self.trange, pot, 'k.', color=tableau20[0])

        ax.set_xlim(0, self.Tplastic)
        ax.set_ylim(-0.01,1.01)
        ax.set_title('difference membrane potential')

    def _zplot(self, ax):
        ax.plot(self.trange2, self.z, label='Z', color=tableau20[0])
        ax.set_xlim(0, self.Tplastic)
        ax.set_title('Z = Sum(r)')

    def _scatter(self, ax):
        #print post-synaptic spikes, spikes leading to reward highlighted

        ax.scatter(np.asarray(self.patternRecord)[:, 1], np.asarray(self.patternRecord)[:, 0], c=np.asarray(self.patternRecord)[:, 2], cmap=plt.get_cmap('winter'), linewidths=0, s=5, label=' ')
        ax.set_xlim(0, self.Tplastic)
        ax.set_ylim(-1, self.nO)

    def _rewardTrace(self, ax):
        #plot low-pass filtered reward trace
        ax.plot(self.trange2, self.totalR, label='mean R', color='darkred')
        ax.fill_between(self.trange, 0, 1, where=self.sessiontime_cut == 0, edgecolor='#FFFFFF', facecolor='#C0C0C0')
        ax.set_title('mean reward')
        ax.set_xlabel('t')

    def _protocol(self, ax):
        #plot tagging protocol
        protocol = [[], []]
        for t in range(len(self.sessiontime_cut)):
            if self.sessiontime[t] == 1:
                protocol[0].append(True)
                protocol[1].append(False)
            elif self.sessiontime[t] == 2:
                protocol[0].append(False)
                protocol[1].append(True)
            else:
                protocol[0].append(False)
                protocol[1].append(False)

        for i in range(2):
            ax.fill_between(self.trange, i, i+1, where=protocol[i], edgecolor=tableau20[i], facecolor=tableau20[i])
        ax.set_xlim(0, self.Tplastic)
        ax.set_title('protocol')
        ax.set_xlabel('t')
        ax.set_yticklabels(['', '', '#1', '', '#2'])


    def fourPatternPlots(self):
        fig1 = plt.figure(figsize=[20, 12], facecolor='#FFFFFF')
        ax1 = fig1.add_axes([.04, .7, .25, .25])
        ax2 = fig1.add_axes([.38, .7, .25, .25])
        ax3 = fig1.add_axes([.7, .7, .25, .25])
        ax4 = fig1.add_axes([.04, .38, .25, .25])
        ax5 = fig1.add_axes([.38, .38, .25, .25])
        ax6 = fig1.add_axes([.7, .38, .25, .25])
        ax7 = fig1.add_axes([.04, .04, .25, .25])
        ax8 = fig1.add_axes([.38, .04, .25, .25])
        ax9 = fig1.add_axes([.7, .04, .25, .25])

        if self.nO > 2:
            self._weightTrajectory(ax1, [self.wF[0][0], self.wF[0][1], self.wF[1][0], self.wF[1][1]], ['$w^f_{11}$', '$w^f_{12}$', '$w^f_{21}$', '$w^f_{22}$'])
            self._weightTrajectory(ax2, [self.wS[0][0], self.wS[0][1], self.wS[1][0], self.wS[1][1]], ['$w^s_{11}$', '$w^s_{12}$', '$w^s_{21}$', '$w^s_{22}$'])
            self._weightTrajectory(ax3, [np.add(self.wF[0][0], self.wS[0][0]), np.add(self.wF[0][1], self.wS[0][1]), np.add(self.wF[1][0], self.wS[1][0]), np.add(self.wF[1][1], self.wS[1][1])], ['$w_{11}$', '$w_{12}$', '$w_{21}$', '$w_{22}$'])

            self._weightTrajectory(ax4, [self.wF[2][2], self.wF[2][3], self.wF[3][2], self.wF[3][3]], ['$w^f_{33}$', '$w^f_{34}$', '$w^f_{43}$', '$w^f_{44}$'])
            self._weightTrajectory(ax5, [self.wS[2][2], self.wS[2][3], self.wS[3][2], self.wS[3][3]], ['$w^s_{33}$', '$w^s_{34}$', '$w^s_{43}$', '$w^s_{44}$'])
            self._weightTrajectory(ax6, [np.add(self.wF[2][2], self.wS[2][2]), np.add(self.wF[2][3], self.wS[2][3]), np.add(self.wF[3][2], self.wS[3][2]), np.add(self.wF[3][3], self.wS[3][3])], ['$w_{33}$', '$w_{34}$', '$w_{43}$', '$w_{44}$'])

            self._weightTrajectory(ax7, [self.wF[0][2], self.wF[0][3], self.wF[3][0], self.wF[3][1]], ['$w^f_{13}$', '$w^f_{14}$', '$w^f_{41}$', '$w^f_{42}$'])
            self._weightTrajectory(ax8, [self.wS[0][2], self.wS[0][3], self.wS[3][0], self.wS[3][1]], ['$w^s_{13}$', '$w^s_{14}$', '$w^s_{41}$', '$w^s_{42}$'])
            self._weightTrajectory(ax9, [np.add(self.wF[0][2], self.wS[0][2]), np.add(self.wF[0][3], self.wS[0][3]), np.add(self.wF[3][0], self.wS[3][0]), np.add(self.wF[3][1], self.wS[3][1])], ['$w_{13}$', '$w_{14}$', '$w_{41}$', '$w_{42}$'])

        else:
            self._weightTrajectory(ax1, [self.wF[0][0], self.wF[0][1], self.wF[0][2], self.wF[0][3]], ['$w^f_{11}$', '$w^f_{12}$', '$w^f_{13}$', '$w^f_{14}$'])
            self._weightTrajectory(ax2, [self.wS[0][0], self.wS[0][1], self.wS[0][2], self.wS[0][3]], ['$w^s_{11}$', '$w^s_{12}$', '$w^s_{13}$', '$w^s_{14}$'])
            self._weightTrajectory(ax3, [np.add(self.wF[0][0], self.wS[0][0]), np.add(self.wF[0][1], self.wS[0][1]), np.add(self.wF[0][2], self.wS[0][2]), np.add(self.wF[0][3], self.wS[0][3])], ['$w_{11}$', '$w_{12}$', '$w_{13}$', '$w_{14}$'])

            self._weightTrajectory(ax4, [self.wF[1][0], self.wF[1][1], self.wF[1][2], self.wF[1][3]], ['$w^f_{21}$', '$w^f_{22}$', '$w^f_{23}$', '$w^f_{24}$'])
            self._weightTrajectory(ax5, [self.wS[1][0], self.wS[1][1], self.wS[1][2], self.wS[1][3]], ['$w^s_{21}$', '$w^s_{22}$', '$w^s_{23}$', '$w^s_{24}$'])
            self._weightTrajectory(ax6, [np.add(self.wF[1][0], self.wS[1][0]), np.add(self.wF[1][1], self.wS[1][1]), np.add(self.wF[1][2], self.wS[1][2]), np.add(self.wF[1][3], self.wS[1][3])], ['$w_{21}$', '$w_{22}$', '$w_{23}$', '$w_{24}$'])

            if self.nI > 4:
                self._weightTrajectory(ax7, [self.wF[0][-1], self.wF[0][-2], self.wF[1][-1], self.wF[1][-2]], ['$w^f_{1-1}$', '$w^f_{1-2}$', '$w^f_{2-1}$', '$w^f_{2-2}$'])
                self._weightTrajectory(ax8, [self.wS[0][-1], self.wS[0][-2], self.wS[1][-1], self.wS[1][-2]], ['$w^s_{1-1}$', '$w^s_{1-2}$', '$w^s_{2-1}$', '$w^s_{2-2}$'])
                self._weightTrajectory(ax9, [np.add(self.wF[0][-1], self.wS[0][-1]), np.add(self.wF[0][-2], self.wS[0][-2]), np.add(self.wF[1][-1], self.wS[1][-1]), np.add(self.wF[1][-2], self.wS[1][-2])], ['$w_{1-1}$', '$w_{1-2}$', '$w_{2-1}$', '$w_{2-2}$'])



        for m in fig1.get_axes():
            self._improveStyle(m)

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        ########################

        fig2 = plt.figure(figsize=[18, 8], facecolor='#FFFFFF')
        ax10 = fig2.add_axes([.07, .55, .25, .4])
        ax11 = fig2.add_axes([.4, .55, .25, .4])
        ax12 = fig2.add_axes([.7, .55, .25, .4])
        ax13 = fig2.add_axes([.07, .07, .25, .4])
        ax14 = fig2.add_axes([.4, .07, .25, .4])
        ax15 = fig2.add_axes([.7, .07, .25, .4])

        if self.nO > 2:
            self._weightSpace(ax10, [(self.wF[0][0], self.wF[0][1]), (self.wF[1][0], self.wF[1][1]), (self.wF[2][0], self.wF[2][1]), (self.wF[3][0], self.wF[3][1])], ['$w_{1j}$', '$w_{2j}$', '$w_{3j}$', '$w_{4j}$'], '$w^f$', 'j=1', 'j=2')
            self._weightSpace(ax11, [(self.wS[0][0], self.wS[0][1]), (self.wS[1][0], self.wS[1][1]), (self.wS[2][0], self.wS[2][1]), (self.wS[3][0], self.wS[3][1])], ['$w_{1j}$', '$w_{2j}$', '$w_{3j}$', '$w_{4j}$'], '$w^s$', 'j=1', 'j=2')
            self._weightSpace(ax12, [(np.add(self.wF[0][0], self.wS[0][0]), np.add(self.wF[0][1], self.wS[0][1])), (np.add(self.wF[1][0], self.wS[1][0]), np.add(self.wF[1][1], self.wS[1][1])), (np.add(self.wF[2][0], self.wS[2][0]), np.add(self.wF[2][1], self.wS[2][1])), (np.add(self.wF[3][0], self.wS[3][0]), np.add(self.wF[3][1], self.wS[3][1]))], ['$w_{1j}$', '$w_{2j}$', '$w_{3j}$', '$w_{4j}$'], '$w^f+w^s$', 'j=2', 'j=1')

            self._weightSpace(ax13, [(self.wF[0][2], self.wF[0][3]), (self.wF[1][2], self.wF[1][3]), (self.wF[2][2], self.wF[2][3]), (self.wF[3][2], self.wF[3][3])], ['$w_{1j}$', '$w_{2j}$', '$w_{3j}$', '$w_{4j}$'], '$w^f$', 'j=3', 'j=4')
            self._weightSpace(ax14, [(self.wS[0][2], self.wS[0][3]), (self.wS[1][2], self.wS[1][3]), (self.wS[2][2], self.wS[2][3]), (self.wS[3][2], self.wS[3][3])], ['$w_{1j}$', '$w_{2j}$', '$w_{3j}$', '$w_{4j}$'], '$w^s$', 'j=3', 'j=4')
            self._weightSpace(ax15, [(np.add(self.wF[0][2], self.wS[0][2]), np.add(self.wF[0][3], self.wS[0][3])), (np.add(self.wF[1][2], self.wS[1][2]), np.add(self.wF[1][3], self.wS[1][3])), (np.add(self.wF[2][2], self.wS[2][2]), np.add(self.wF[2][3], self.wS[2][3])), (np.add(self.wF[3][2], self.wS[3][2]), np.add(self.wF[3][3], self.wS[3][3]))], ['$w_{1j}$', '$w_{2j}$', '$w_{3j}$', '$w_{4j}$'], '$w^f+w^s$', 'j=4', 'j=3')

        if self.nO == 2 and self.nI > 4:
            self._weightSpace(ax10, [(self.wF[0][0], self.wF[0][-1]), (self.wF[1][0], self.wF[1][-1])], ['$w_{1j}$', '$w_{2j}$'], '$w^f$', 'j=1', 'j=-1')
            self._weightSpace(ax11, [(self.wS[0][0], self.wS[0][-1]), (self.wS[1][0], self.wS[1][-1])], ['$w_{1j}$', '$w_{2j}$'], '$w^s$', 'j=1', 'j=-1')
            self._weightSpace(ax12, [(np.add(self.wF[0][0], self.wS[0][0]), np.add(self.wF[0][-1], self.wS[0][-1])), (np.add(self.wF[1][0], self.wS[1][0]), np.add(self.wF[1][-1], self.wS[1][-1]))], ['$w_{1j}$', '$w_{2j}$'], '$w^f+w^s$', 'j=1', 'j=-1')


        for m in fig2.get_axes():
            self._improveStyle(m)

        ########################

        fig3 = plt.figure(figsize=[18, 12], facecolor='#FFFFFF')
        ax16 = fig3.add_axes([.05, .75, .38, .19])
        ax17 = fig3.add_axes([.05, .52, .38, .19])
        ax18 = fig3.add_axes([.05, .29, .38, .19])
        ax19 = fig3.add_axes([.05, .03, .38, .19])
        ax20a = fig3.add_axes([.6, .75, .38, .19])
        ax20b = fig3.add_axes([.6, .52, .38, .19])
        ax20c = fig3.add_axes([.6, .29, .38, .19])
        ax20d = fig3.add_axes([.6, .04, .38, .19])

        self._activationFunction(ax16)

        self._scatter(ax17)

        self._rewardTrace(ax18)

        self._zplot(ax19)

        if self.nO > 4:
            self._membranePotential(ax20a, 0, [0, 1, 2, 3, 4])
            self._membranePotential(ax20b, 1, [0, 1, 2, 3, 4])
            self._membranePotential(ax20c, 2, [0, 1, 2, 3, 4])
            self._membranePotential(ax20d, 3, [0, 1, 2, 3, 4])
        else:
            self._membranePotential(ax20a, 0, [0, 1])
            self._membranePotential(ax20b, 1, [0, 1])
            self._membranePotential(ax20c, 2, [0, 1])
            self._membranePotential(ax20d, 3, [0, 1])

        for m in fig3.get_axes():
            self._improveStyle(m)

    def twoPatternPlots(self):
        fig1 = plt.figure(figsize=[20, 12], facecolor='#FFFFFF')
        ax1 = fig1.add_axes([.04, .7, .25, .25])
        ax2 = fig1.add_axes([.38, .7, .25, .25])
        ax3 = fig1.add_axes([.7, .7, .25, .25])
        ax4 = fig1.add_axes([.04, .38, .25, .25])
        ax5 = fig1.add_axes([.38, .38, .25, .25])
        ax6 = fig1.add_axes([.7, .38, .25, .25])
        ax7 = fig1.add_axes([.04, .04, .25, .25])
        ax8 = fig1.add_axes([.38, .04, .25, .25])
        ax9a = fig1.add_axes([.7, .2, .25, .12])
        ax9b = fig1.add_axes([.7, .04, .25, .12])

        self._weightTrajectory(ax1, [self.wF[0][0], self.wF[0][1], self.wF[1][0], self.wF[1][1]], ['$w^f_{11}$', '$w^f_{12}$', '$w^f_{21}$', '$w^f_{22}$'])
        self._weightTrajectory(ax2, [self.wS[0][0], self.wS[0][1], self.wS[1][0], self.wS[1][1]], ['$w^s_{11}$', '$w^s_{12}$', '$w^s_{21}$', '$w^s_{22}$'])
        self._weightTrajectory(ax3, [np.add(self.wF[0][0], self.wS[0][0]), np.add(self.wF[0][1], self.wS[0][1]), np.add(self.wF[1][0], self.wS[1][0]), np.add(self.wF[1][1], self.wS[1][1])], ['$w_{11}$', '$w_{12}$', '$w_{21}$', '$w_{22}$'])

        if self.nO > 3:
            self._weightSpace(ax4, [(self.wF[0][0], self.wF[0][1]), (self.wF[1][0], self.wF[1][1]), (self.wF[2][0], self.wF[2][1]), (self.wF[3][0], self.wF[3][1])], ['$w_{1j}$', '$w_{2j}$', '$w_{3j}$', '$w_{4j}$'], '$w^f$', 'j=1', 'j=2')
            self._weightSpace(ax5, [(self.wS[0][0], self.wS[0][1]), (self.wS[1][0], self.wS[1][1]), (self.wS[2][0], self.wS[2][1]), (self.wS[3][0], self.wS[3][1])], ['$w_{1j}$', '$w_{2j}$', '$w_{3j}$', '$w_{4j}$'], '$w^s$', 'j=1', 'j=2')
            self._weightSpace(ax6, [(np.add(self.wF[0][0], self.wS[0][0]), np.add(self.wF[0][1], self.wS[0][1])), (np.add(self.wF[1][0], self.wS[1][0]), np.add(self.wF[1][1], self.wS[1][1])), (np.add(self.wF[2][0], self.wS[2][0]), np.add(self.wF[2][1], self.wS[2][1])), (np.add(self.wF[3][0], self.wS[3][0]), np.add(self.wF[3][1], self.wS[3][1]))], ['$w_{1j}$', '$w_{2j}$', '$w_{3j}$', '$w_{4j}$'], '$w^f+w^s$', 'j=1', 'j=2')
        else:
            self._weightSpace(ax4, [(self.wF[0][0], self.wF[0][1]), (self.wF[1][0], self.wF[1][1])], ['$w_{1j}$', '$w_{2j}$'], '$w^f$', 'j=1', 'j=2')
            self._weightSpace(ax5, [(self.wS[0][0], self.wS[0][1]), (self.wS[1][0], self.wS[1][1])], ['$w_{1j}$', '$w_{2j}$'], '$w^s$', 'j=1', 'j=2')
            self._weightSpace(ax6, [(np.add(self.wF[0][0], self.wS[0][0]), np.add(self.wF[0][1], self.wS[0][1])), (np.add(self.wF[1][0], self.wS[1][0]), np.add(self.wF[1][1], self.wS[1][1]))], ['$w_{1j}$', '$w_{2j}$'], '$w^f+w^s$', 'j=1', 'j=2')


        #self._activationFunction(ax7)
        self._zplot(ax7)

        self._rewardTrace(ax8)
        #self._diffmembranePotential(ax8, [0,1])

        if self.nO > 2:
            self._membranePotential(ax9a, 0, [0, 1, 2])
            self._membranePotential(ax9b, 1, [0, 1, 2])
        else:
            self._membranePotential(ax9a, 0, [0, 1])
            self._membranePotential(ax9b, 1, [0, 1])

        for m in fig1.get_axes():
            self._improveStyle(m)

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

    def taggingPlots(self):
        fig1 = plt.figure(figsize=[20, 12], facecolor='#FFFFFF')
        ax1 = fig1.add_axes([.04, .7, .25, .25])
        ax2 = fig1.add_axes([.38, .7, .25, .25])
        ax3 = fig1.add_axes([.7, .7, .25, .25])
        ax4 = fig1.add_axes([.04, .38, .25, .25])
        ax7 = fig1.add_axes([.04, .04, .25, .25])
        ax6 = fig1.add_axes([.7, .38, .25, .25])
        ax9 = fig1.add_axes([.7, .04, .25, .25])


        thresline = np.ones(len(self.wF[0][0]))*self.thres2
        self._weightTrajectory(ax1, [self.wF[0][0], self.wF[0][1], thresline], ['$w^f_{11}$', '$w^f_{12}$', 'thres'])
        self._weightTrajectory(ax2, [self.wS[0][0], self.wS[0][1]], ['$w^s_{11}$', '$w^s_{12}$'])
        self._weightTrajectory(ax3, [np.add(self.wF[0][0], self.wS[0][0]), np.add(self.wF[0][1], self.wS[0][1])], ['$w_{11}$', '$w_{12}$'])

        thresline = np.ones(len(self.wF[0][0]))*self.thres1
        dummy = np.zeros(len(self.wF[0][0]))
        self._weightTrajectory(ax4, [self.vMean[0], dummy, thresline], ['$v_1$ (mean)', '', 'thres'])

        self._protocol(ax7)

        self._membranePotential(ax6, 0, [0])
        self._membranePotential(ax9, 1, [0])

        for m in fig1.get_axes():
            self._improveStyle(m)

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()