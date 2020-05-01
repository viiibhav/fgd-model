# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:57:34 2020

@author: vdabadgh
"""

from __future__ import division, print_function

from droplet_reduced_dim_12 import m
from WFGD_data import (i, sp_all, d, z_sp, c, gam, Keq_const, Diff,
                        m_limestone, rho_l)
from pyomo.environ import (ConcreteModel, Set, Param, Var,
                            Expression, Constraint, Objective,
                            NonNegativeReals, exp, log, value,
                            TransformationFactory, minimize, Suffix)
from pyomo.opt import (SolverFactory, SolverStatus, TerminationCondition,
                        ProblemFormat)
from pyomo.dae import DerivativeVar, ContinuousSet
# from pyomo.contrib.pynumero.interfaces import PyomoNLP

import matplotlib.pyplot as plt
import numpy as np
pi = np.pi
# import cloudpickle as pickle

from droplet_reduced_dim_12 import m, opt
