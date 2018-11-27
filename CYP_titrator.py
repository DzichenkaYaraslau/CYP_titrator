#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:01:40 2018

@author: botan
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid.inset_locator import inset_axes

def TightBinding(C, Amax, E, Kd):
    return Amax * (C + E + Kd - np.sqrt(np.power(C + E + Kd, 2) - 4 * E * C))/(2 * E)

fileName = 'CYP21_M3.csv'
fIn = open(fileName, 'r')
lines = fIn.readlines()
fIn.close()
lmin = 349
lmax = 500
step = 1
wl_range = range(lmax, lmin, -step)
vol_range = [float(i) for i in lines[0].strip('\n').split(',,')[:-1]]
y = len(lines[0].strip('\n').split(',,'))-1
x = len(wl_range)
data = np.zeros(x*y).reshape(x, y)

zeroMax = [float(item) for item in lines[2].strip('\n').split(',')[1::2]]
for i in wl_range:
    data[lmax-i] = [float(item)-zeroMax[j] for j, item in enumerate(lines[lmax-i+2].strip('\n').split(',')[1::2])]

df = pd.DataFrame(data, index=wl_range, columns=vol_range)
finC = df[vol_range[-1]]
#print(finC)
maxind = lmin + max(range(len(finC)), key = lambda x: finC[x + lmin+1])
minind = lmin + min(range(len(finC)), key = lambda x: finC[x + lmin+1])
DA = np.zeros(len(vol_range))
for i, item in enumerate(df):
    DA[i] = df[item][maxind]-df[item][minind]
Vsum =0
conc = np.zeros(len(vol_range))
for i in range(1, len(vol_range)):
    if (vol_range[i]-vol_range[i-1] > 10) and (vol_range[i]-vol_range[i-1] < 100):
        Vsum += (vol_range[i]-vol_range[i-1])/10
    elif vol_range[i]-vol_range[i-1] > 100:
        Vsum += (vol_range[i]-vol_range[i-1])/100
    else:
        Vsum = vol_range[i]
    conc[i] = 100/(2000+Vsum)*vol_range[i]

popt, pcov = curve_fit(TightBinding, conc, DA, (max(DA), 1., 40), bounds = ((0, 0.99, 0), (np.inf, 1, np.inf)))   
Amax, E, Kd = popt

print('Amax={0}\nE={1}\nKd={2}'.format(*tuple(popt)), float(np.sqrt(np.diag(pcov))[2]))
fig, ax1 = plt.subplots()
fig.tight_layout()
ax1.plot(conc, DA, 'ro', conc, TightBinding(conc, *popt))
ax1.set_title(fileName[:-4])
ax1.set_xlabel('[L], uM')
ax1.set_ylabel(r'$\Delta$A')
#ax1.text('left top')

ax1_inset = inset_axes(ax1, width="40%", height="40%", loc=4, borderpad=3.5)
ax1_inset.plot(df)
ax1_inset.set_xlim((lmin, lmax))
ax1_inset.set_xlabel(r'$\lambda$, nm')
ax1_inset.set_ylabel('A')
plt.tight_layout()

fig.savefig(fileName[:-4]+".png")