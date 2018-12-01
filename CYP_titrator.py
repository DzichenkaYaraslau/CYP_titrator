#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:01:40 2018

@author: botan
"""

import numpy as np
from lmfit import minimize, Parameters, Parameter, fit_report
import pandas as pd
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import argparse

def getParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default=None)
    parser.add_argument('-b', '--lmin', default=349)
    parser.add_argument('-r', '--lmax', default=500) 
    parser.add_argument('-s', '--step', default=1)     
    parser.add_argument('-E', '--enzyme', default=1) 
    parser.add_argument('-e', '--e_vary', default=False) 
    parser.add_argument('-a', '--amax_vary', default=True) 
    parser.add_argument('-k', '--kd_vary', default=True) 
    return parser

def TightBinding(params, C, DA):
    Amax = params['Amax'].value
    E = params['E'].value
    Kd = params['Kd'].value
    model = Amax * (C + E + Kd - np.sqrt((C + E + Kd)**2 - 4 * E * C))/(2 * E)
    return model-DA

parser = getParameters()
namespace = parser.parse_args(sys.argv[1:])
#print(namespace)
fileName = namespace.filename#'CYP7A1_LjG16-8M.csv'
fIn = open(fileName, 'r')
lines = fIn.readlines()
fIn.close()
lmin = namespace.lmin
lmax = namespace.lmax
step = namespace.step
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

maxind = lmin + max(range(len(finC)), key = lambda x: finC[x + lmin+1])
minind = lmin + min(range(len(finC)), key = lambda x: finC[x + lmin+1])
DA = np.zeros(len(vol_range))
for i, item in enumerate(df):
    DA[i] = df[item][maxind]-df[item][minind]
Vsum =0
C = np.zeros(len(vol_range))
for i in range(1, len(vol_range)):
    if (vol_range[i]-vol_range[i-1] > 10) and (vol_range[i]-vol_range[i-1] < 100):
        Vsum += (vol_range[i]-vol_range[i-1])/10
    elif vol_range[i]-vol_range[i-1] > 100:
        Vsum += (vol_range[i]-vol_range[i-1])/100
    else:
        Vsum = vol_range[i]
    C[i] = 100/(2000+Vsum)*vol_range[i]

params = Parameters()
params['Amax'] = Parameter(name='Amax', value=max(DA), vary = namespace.amax_vary)
params['E'] = Parameter(name='E', value=namespace.enzyme, vary = namespace.e_vary)
params['Kd'] = Parameter(name='Kd', value=1, vary = namespace.kd_vary)
params.update_constraints()
result = minimize(TightBinding, params, args=(C,DA))
final = DA + result.residual
print(fit_report(result.params))
fig, ax1 = plt.subplots()
fig.tight_layout()
ax1.plot(C, DA, 'ro')
ax1.plot(C, final, 'b')
ax1.text(0.0, 1.0, fit_report(result.params), fontsize=8, ha='left', va='top', transform=ax1.transAxes)
ax1.set_title(fileName[:-4])
ax1.set_xlabel('[L], uM')
ax1.set_ylabel(r'$\Delta$A')

ax1_inset = inset_axes(ax1, width="40%", height="40%", loc=4, borderpad=3.5)
ax1_inset.plot(df)
ax1_inset.set_xlim((lmin, lmax))
ax1_inset.set_xlabel(r'$\lambda$, nm')
ax1_inset.set_ylabel('A')
plt.tight_layout()

fig.savefig(fileName[:-4]+".png")