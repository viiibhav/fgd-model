# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:20:15 2020

@author: vdabadgh
"""

from __future__ import division, print_function

from pyomo.environ import (ConcreteModel, Set, Param, Var,
                           Expression, Constraint, Objective,
                           NonNegativeReals, exp, log, value,
                           TransformationFactory, minimize, Suffix)
from pyomo.opt import (SolverFactory, SolverStatus, TerminationCondition,
                       ProblemFormat)
from pyomo.dae import DerivativeVar, ContinuousSet
from pyomo.contrib.pynumero.interfaces import PyomoNLP

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import seaborn as sns
sns.set_style("whitegrid")
pi = np.pi
import cloudpickle as pickle


"""
Scrubber Model
"""


m = ConcreteModel()

# =============================================================================
# Sets
# =============================================================================
z = list(np.linspace(0, 10))
m.z = Set(initialize=z)


# =============================================================================
# Params
# =============================================================================
g = 9.81  # m/s**2
R = 8.314  # J/mol-K
H = 6.38e-03  # kmol/m**3/Pa
mu_g = 1.95e-05  # Pa-s
M_air = 29  # g/mol
M_SO2 = 64  # g/mol
V_air = 20.1  # cm**3/mol
V_SO2 = 41.1  # cm**3/mol
K_hc = 79.73
K_c = 1.8e-05
K_w = 5.58e-14
K_ha = 5.4e-06
K_a1 = 7.94e-03
K_a2 = 4.4e-08
T = 323.15  # K
p = 101325  # Pa
rho_p = 1003.6 / 5  # [kg/m**3]
rho_g = 0.9  # [kg/m**3]
ug = 3.0  # m/s
up0 = 7.77  # m/s
dp = 1.8e-03  # m
G = 456120.3 / 3600  # m**3/s
L = 3 * G
logN = 13.680 - 4987 / T
N = 10**(logN)  # or e^logN? Check.  # reference Johnstone I&EC 1937 pp 1396.

y_in = 0.001  # guess
y_out = 0.0001
cH0 = 1e-05


# droplet velocity 
def _dupdt(up, t):
    Re = dp * abs(up - ug) * rho_g / mu_g
    Cdrag = 24 / Re * (1 + 0.125 * R**0.72)
    return g * (rho_p - rho_g) / rho_p - \
        3 / 4 * (rho_g * (up - ug)**2 * Cdrag) / (rho_p * dp)


t = np.linspace(0, 8)
up = odeint(_dupdt, up0, t)
plt.figure(1)
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.plot(t, up)


# distribution coefficient
def lam(c, i):
    den = c**2 + K_a1 * c + K_a1 * K_a2
    if i == 0:
        return c**2 / den
    elif i == 1:
        return K_a1 * c / den
    else:
        return K_a1 * K_a2 / den


pH_range = np.linspace(0.2, 9.2)
cH = 10**(-pH_range)
plt.figure(2)
plt.plot(pH_range, lam(cH, 0), linestyle="-", marker="s", label=r"H$_2$SO$_3$")
plt.plot(pH_range, lam(cH, 1), linestyle="-", marker="D", label=r"HSO$_3^-$")
plt.plot(pH_range, lam(cH, 2), linestyle="-", marker="^", label=r"SO$_3^{2-}$")
plt.legend()


# y_eqm = p_star / p
dz = 0.1  # m
a = 1 / 3600 * L / (np.pi / 6 * dp**3) * np.pi * dp**2 * dz / up0
D_SO2 = (9.86e-03 * T**1.75 * (1 / M_air + 1 / M_SO2)**0.5) / \
        (p * (V_air**(1 / 3) + V_SO2**(1 / 3))**2)
Sc = mu_g / (rho_g * D_SO2)
Re = dp * abs(up - ug) * rho_g / mu_g
Sh = 2 + 0.552 * Re**0.5 * Sc**(1 / 3)
ky = Sh * D_SO2 * H * p / dp
cSO2 = 2072.24e-03 / M_SO2 * 1e-03  # mol/m**L
cSO4 = cSO2# * 1000


def p_star(cH):
    num = cH**3 - 2 * cSO4 * cH**2 - K_w * cH
    den1 = K_ha * K_a1 * cH + 2 * K_ha * K_a1 * K_a2
    den2 = 1 - 133.322 * N * K_a2 * K_hc * K_c / K_w
    return num / (den1 * den2)


p0SO4 = p_star(cH[25])  # cH such that pH=5 approx


def f(cH):
    I0 = K_ha * p0SO4 * (1 + K_a1 / cH0 + (K_a1 * K_a2) / cH0**2)
    expr = 0
    num1 = cH**2 - 2 * cSO4 - K_w
    den1 = G * cH * (1 - y_in) * (K_ha * K_a1 * cH + 2 * K_ha * K_a1 * K_a2) / L
    num2 = K_ha * cH**2 + K_ha * K_a1 * cH + K_ha * K_a1 * K_a2
    den2 = 1 - 133.322 * N * K_hc * K_c * K_a2 / K_w
    expr += num1 * num2 / (den1 * den2)
    expr -= I0 * L / (G * (1 - y_in))
    expr += y_out / (1 - y_out)
    return expr


def dfdcH(cH):
    A = G / L * (1 - y_in)
    B = 2 * cSO4 + K_w
    K1 = K_ha * K_a1
    K2 = 2 * K_ha * K_a1 * K_a2
    alpha = 1 - 133.332 * N * K_a2 * K_hc * K_c / K_w
    num1 = cH**2 - B
    den1 = A * cH * (K1 * cH + K2)
    g1 = num1 / den1
    dg1 = (den1 * 2 * cH - num1 * (A * cH * K1 + A * (K1 * cH + K2))) / den1**2
    g2 = (K_ha * cH**2 + K1 * cH + K2 / 2) / alpha
    dg2 = (2 * K_ha * cH + K1) / alpha
    return g1 * dg2 + g2 * dg1


def dcHdz(cH, z):
    y = f(cH) / (1 + f(cH))
    return ky * a * (y - y_in) / (G * (1 - y_in)) * 1 / dfdcH(cH)








