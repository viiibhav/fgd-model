# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:23:55 2020

@author: vdabadgh
"""

from __future__ import division, print_function

from WFGD_data import (i, sp_all, d, z_sp, c, gam, Keq_const, Diff,
                       m_limestone, rho_l)
from pyomo.environ import (ConcreteModel, Set, Param, Var, Block,
                           Expression, Constraint, ConstraintList, Objective,
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


"""
Modeling SO2 absorption -- Brogren and Karlsson (1997)
All units in cm, cm**3 etc.
"""


m = ConcreteModel()

# =============================================================================
# Sets
# =============================================================================
# Radial coordinate
r = d / 2
nfe = 50
decimals = 10
dx = round(r / nfe, decimals)
r = round(dx * nfe, decimals)
x = list(np.round(np.linspace(0, r, num=nfe+1), decimals=decimals))
m.x = ContinuousSet(initialize=x)

# Species
spc = [sp for sp in sp_all if sp != "CO2" and sp != "SO2"]
i.remove("CaHSO3+")
i.remove("MgHSO3+")
i.remove("MgSO3")
i.remove("MgSO4")
i.remove("Mg2+")
i.remove("MgHCO3+")
i.remove("Cl-")
i.remove("SO42-")
i.remove("CaSO4")

m.i = Set(initialize=i)

# Eqm species
eq_i = ["SO2", "HSO3-", "HCO3-", "CaSO3", "CaCO3", "CaHCO3+"]
m.eq_i = Set(initialize=eq_i)

alg_vars = ["SO2", "Ca2+", "OH-", "SO32-", "HSO3-", "HCO3-", "CO32-"]
diff_vars = [sp for sp in i if sp not in alg_vars]

sulfite_species = [sp for sp in i if "SO" in sp]
carbonate_species = [sp for sp in i if "CO" in sp]
calcium_species = [sp for sp in i if "Ca" in sp]
charged_species = [sp for sp in i if "+" in sp or "-" in sp]


# =============================================================================
# Params
# =============================================================================
T = 25 + 273.15  # [K]
R = 8.314        # [J/mol-K]

# Equilibrium constants (Radian corp. data, 1970)
# [Keq] = mol/L --> mol/cm**3. [Kw] = (mol/L)**2 --> (mol/cm**3)**2
Keq_all = {sp: 10**(-vals[0] / T - vals[1] * np.log10(T) - vals[2] * T +
                    vals[3]) for sp, vals in Keq_const.items()}
Keq = {sp: k for sp, k in Keq_all.items() if sp in eq_i and sp != "SO2"}
Keq["SO2"] = Keq_all["H2SO3"]


"""
Using new K values from other literature:
K[SO2] and K[HSO3-] from ammonia based scrubber paper (2011)
K[CaCO3] and K[CaHCO3+] from Jacobson and Langmuir (1973)
K[HCO3-] from Roy et al. (1993)
K[CaSO3] from Frydman et al. (1958)
"""

m.Keq = Param(m.eq_i, initialize=Keq)
Kw = Keq_all["H2O"]
m.Kw = Param(initialize=Kw)

# Diffusivity [m**2/s]
m.Diff = Param(m.i, rule=(lambda m, i: Diff[i]))

# charge
z_sp["CO2"], z_sp["SO2"] = 0, 0
m.z = Param(m.i, rule=(lambda m, i: z_sp[i]))

# partial pressures [Pa]  # Brogren & Karlsson
p, p_i = {}, {}
p["SO2"] = 80
p["CO2"] = 10    # ?
p_i["SO2"] = 20  # ?
p_i["CO2"] = 2   # ?
m.gases = Set(initialize=["SO2", "CO2"])
m.p = Param(m.gases, initialize=p)
m.p_i = Param(m.gases, initialize=p)

# mass transfer coeff [mol/(m**2 s Pa)]
kG = {}
kG["SO2"] = 5e-05  # Brogren & Karlsson
kG["CO2"] = 5e-05


# =============================================================================
# Vars
# =============================================================================
#csol = pickle.load(open("csol_50.pkl", "rb"))
#csol_avg = {sp: np.mean(list(csol[sp].values())) for sp in m.i}
#def _c_init(m, i, x):
#    return csol_avg[i]
#m.c = Var(m.i, m.x, initialize=_c_init)
m.c = Var(m.i, m.x, initialize=1e-03)
m.u = Var(m.i, m.x, initialize=1e-06)
m.v = Var(m.i, m.x, initialize=1e-03)
m.dcdx = DerivativeVar(m.c, wrt=(m.x), initialize=1e-03)
m.dudx = DerivativeVar(m.u, wrt=(m.x), initialize=0.0)


# =============================================================================
# Model
# =============================================================================
# Relative supersaturation
def _RS_CaCO3(m, x):
    return m.c["Ca2+", x] * m.c["CO32-", x] / Keq_all["CaCO3(s)"]


m.RS_CaCO3 = Expression(m.x, rule=_RS_CaCO3)


def _RS_CaSO3(m, x):
    return m.c["Ca2+", x] * m.c["SO32-", x] / Keq_all["CaSO3(s)"]


m.RS_CaSO3 = Expression(m.x, rule=_RS_CaSO3)


# =============================================================================
# Finite rate reactions occuring in the slurry

# Hydrolysis of CO2
# Forward rate of reaction (Pinsent et al. 1956) [1/s]
logk1_CO2 = 329.85 - 110.541 * np.log10(T) - 17265.5 / T
k1_CO2 = 10**(logk1_CO2)


def _r_CO2(m, x):
    keq_CO2 = Keq_all["H2CO3"]
    return k1_CO2 * m.c["CO2", x] * (m.c["H+", x] * m.c["HCO3-", x] /
                                     (keq_CO2 * m.c["CO2", x]) - 1)


m.r_CO2 = Expression(m.x, rule=_r_CO2)


# =============================================================================
# Limestone dissolution
def _rd_CaCO3(m, x):
#    if x == r:  # 0 to skip at surface
    if x == 0:
        return Expression.Skip
    else:
        xi = 0
        for k in ["HCO3-", "CO32-", "CaCO3"]:
            xi += -m.Diff[k] * m.u[k, x]
        return 6 * m_limestone / (d * rho_l) * xi


m.rd_CaCO3 = Expression(m.x, rule=_rd_CaCO3)


# =============================================================================
# Calcium sulfite crystallization
def _rc_CaSO3(m, x):
    return 0


m.rc_CaSO3 = Expression(m.x, rule=_rc_CaSO3)


# =============================================================================
# Henry's constants (conc * H = pressure)
den = 0.011843 * R * T * np.exp(3100 * (1 / T - 1 / 298.15))
H_SO2 = 1 / den
H_SO2 *= 1e+06  # convert to [Pa / (mol/cm**3)]
m.H_SO2 = Param(initialize=H_SO2)

rho_CO2 = 1.98     # [kg/m**3]  # STP
H_CO2 = 1 / 0.035  # [bar/(mol/kg)]  # NIST
H_CO2 *= 10**5     # [Pa / (mol/kg)]
H_CO2 /= rho_CO2   # [Pa / (mol/m**3)]
H_CO2 *= 1e+06     # convert to [Pa / (mol/cm**3)]
m.H_CO2 = Param(initialize=H_CO2)


# =============================================================================
# Equilibrium reactions
def _water_diss(m, x):
    return m.c["H+", x] * m.c["OH-", x] == m.Kw
# m.water_diss = Constraint(m.x, rule=_water_diss)


def _H2SO3_diss(m, x):  # use Keq[H2SO3]
    return m.Keq["SO2"] * m.c["SO2", x] == m.c["H+", x] * m.c["HSO3-", x]
# m.H2SO3_diss = Constraint(m.x, rule=_H2SO3_diss)


def _HSO3_diss(m, x):
    return m.Keq["HSO3-"] * m.c["HSO3-", x] == m.c["H+", x] * m.c["SO32-", x]
# m.HSO3_diss = Constraint(m.x, rule=_HSO3_diss)


def _HCO3_diss(m, x):
    return m.Keq["HCO3-"] * m.c["HCO3-", x] == m.c["H+", x] * m.c["CO32-", x]
# m.HCO3_diss = Constraint(m.x, rule=_HCO3_diss)


def _CaSO3_diss(m, x):
    return m.Keq["CaSO3"] * m.c["CaSO3", x] == m.c["Ca2+", x] * m.c["SO32-", x]
# m.CaSO3_diss = Constraint(m.x, rule=_CaSO3_diss)


def _CaCO3_diss(m, x):
    return m.Keq["CaCO3"] * m.c["CaCO3", x] == m.c["Ca2+", x] * m.c["CO32-", x]
# m.CaCO3_diss = Constraint(m.x, rule=_CaCO3_diss)


def _CaHCO3_diss(m, x):
    return m.Keq["CaHCO3+"] * m.c["CaHCO3+", x] == m.c["Ca2+", x] * m.c["HCO3-", x]
# m.CaHCO3_diss = Constraint(m.x, rule=_CaHCO3_diss)


# =============================================================================
# Twice-differentiated equations
def _water_diss_der2(m, x):
#    if x == 0:
##        return (m.u["H+", x+dx] - m.u["H+", x]) / dx * m.c["OH-", x] + \
##            m.c["H+", x] * (m.u["OH-", x+dx] - m.u["OH-", x]) / dx + \
##            2 * m.u["H+", x] * m.u["OH-", x] == 0
#    else:
    return m.v["H+", x] * m.c["OH-", x] + \
        m.c["H+", x] * m.v["OH-", x] + \
        2 * m.u["H+", x] * m.u["OH-", x] == 0
# m.water_diss_der2 = Constraint(m.x, rule=_water_diss_der2)


def _H2SO3_diss_der2(m, x):  # use Keq[H2SO3]
#    if x == r:  # SURFACE CONDITION FOR SO2!
##    if x == 0:
##        return m.Keq["SO2"] * (m.u["SO2", x+dx] - m.u["SO2", x]) / dx == \
##            (m.u["H+", x+dx] - m.u["H+", x]) / dx * m.c["HSO3-", x] + \
##            m.c["H+", x] * (m.u["HSO3-", x+dx] - m.u["HSO3-", x]) / dx + \
##            2 * m.u["H+", x] * m.u["HSO3-", x]
#    else:
    return m.Keq["SO2"] * m.v["SO2", x] == \
        m.v["H+", x] * m.c["HSO3-", x] + \
        m.c["H+", x] * m.v["HSO3-", x] + \
        2 * m.u["H+", x] * m.u["HSO3-", x]
# m.H2SO3_diss_der2 = Constraint(m.x, rule=_H2SO3_diss_der2)


def _HSO3_diss_der2(m, x):
#    if x == 0:
##        return m.Keq["HSO3-"] * (m.u["HSO3-", x+dx] - m.u["HSO3-", x]) / dx == \
##            (m.u["H+", x+dx] - m.u["H+", x]) / dx * m.c["SO32-", x] + \
##            m.c["H+", x] * (m.u["SO32-", x+dx] - m.u["SO32-", x]) / dx + \
##            2 * m.c["H+", x] * m.u["SO32-", x]
#    else:
    return m.Keq["HSO3-"] * m.v["HSO3-", x] == \
        m.v["H+", x] * m.c["SO32-", x] + \
        m.c["H+", x] * m.v["SO32-", x] + \
        2 * m.c["H+", x] * m.u["SO32-", x]
# m.HSO3_diss_der2 = Constraint(m.x, rule=_HSO3_diss_der2)


def _HCO3_diss_der2(m, x):
#    if x == 0:
##        return m.Keq["HCO3-"] * (m.u["HCO3-", x+dx] - m.u["HCO3-", x]) / dx == \
##            (m.u["H+", x+dx] - m.u["H+", x]) / dx * m.c["CO32-", x] + \
##            m.c["H+", x] * (m.u["CO32-", x+dx] - m.u["CO32-", x]) / dx + \
##            2 * m.u["H+", x] * m.u["CO32-", x]
#    else:
    return m.Keq["HCO3-"] * m.v["HCO3-", x] == \
        m.v["H+", x] * m.c["CO32-", x] + \
        m.c["H+", x] * m.v["CO32-", x] + \
        2 * m.u["H+", x] * m.u["CO32-", x]
# m.HCO3_diss_der2 = Constraint(m.x, rule=_HCO3_diss_der2)


def _CaSO3_diss_der2(m, x):
#    if x == 0:
##        return m.Keq["CaSO3"] * (m.u["CaSO3", x+dx] - m.u["CaSO3", x]) / dx == \
##            (m.u["Ca2+", x+dx] - m.u["Ca2+", x]) / dx * m.c["SO32-", x] + \
##            m.c["Ca2+", x] * (m.u["SO32-", x+dx] - m.u["SO32-", x]) / dx + \
##            2 * m.u["Ca2+", x] * m.u["SO32-", x]
#    else:
    return m.Keq["CaSO3"] * m.v["CaSO3", x] == \
        m.v["Ca2+", x] * m.c["SO32-", x] + \
        m.c["Ca2+", x] * m.v["SO32-", x] + \
        2 * m.u["Ca2+", x] * m.u["SO32-", x]
# m.CaSO3_diss_der2 = Constraint(m.x, rule=_CaSO3_diss_der2)


def _CaCO3_diss_der2(m, x):
#    if x == 0:
##        return m.Keq["CaCO3"] * (m.u["CaCO3", x+dx] - m.u["CaCO3", x]) / dx == \
##            (m.u["Ca2+", x+dx] - m.u["Ca2+", x]) / dx * m.c["CO32-", x] + \
##            m.c["Ca2+", x] * (m.u["CO32-", x+dx] - m.u["CO32-", x]) / dx + \
##            2 * m.u["Ca2+", x] * m.u["CO32-", x]
#    else:
    return m.Keq["CaCO3"] * m.v["CaCO3", x] == \
        m.v["Ca2+", x] * m.c["CO32-", x] + \
        m.c["Ca2+", x] * m.v["CO32-", x] + \
        2 * m.u["Ca2+", x] * m.u["CO32-", x]
# m.CaCO3_diss_der2 = Constraint(m.x, rule=_CaCO3_diss_der2)


def _CaHCO3_diss_der2(m, x):
#    if x == 0:
##        return m.Keq["CaHCO3+"] * (m.u["CaHCO3+", x+dx] - m.u["CaHCO3+", x]) / dx == \
##            (m.u["Ca2+", x+dx] - m.u["Ca2+", x]) / dx * m.c["HCO3-", x] + \
##            m.c["Ca2+", x] * (m.u["HCO3-", x+dx] - m.u["HCO3-", x]) / dx + \
##            2 * m.u["Ca2+", x] * m.u["HCO3-", x]
#    else:
    return m.Keq["CaHCO3+"] * m.v["CaHCO3+", x] == \
        m.v["Ca2+", x] * m.c["HCO3-", x] + \
        m.c["Ca2+", x] * m.v["HCO3-", x] + \
        2 * m.u["Ca2+", x] * m.u["HCO3-", x]
# m.CaHCO3_diss_der2 = Constraint(m.x, rule=_CaHCO3_diss_der2)


# =============================================================================
# Define u = dcdx
def _reform_con(m, i, x):
    if i in diff_vars:
#        if x == 0:  # r for ????
        if x == r:
#            return (m.c[i, x+dx] - m.c[i, x]) / dx == 0.0
            return m.c[i, x] == 0.0005
#            return (m.c[i, x] - m.c[i, x-dx]) / dx == 0.0
        else:
            return m.u[i, x] == m.dcdx[i, x]
    else:
        return Constraint.Skip
# m.reform_con = Constraint(m.i, m.x, rule=_reform_con)


#for comp in diff_vars:
#    m.reform_con[comp, 0].deactivate()
#    m.reform_con.add([comp, 0], expr=())


# =============================================================================
# Define dudx = v
def _reform_con_2(m, i, x):
    if i == "SO2" or i == "CO2":
#        if x == 0:
        if x == r:
            return m.u[i, x] == kG[i] * (p[i] - p_i[i])
        else:
            return m.dudx[i, x] == m.v[i, x]
    else:
        if x == 0:
            return m.u[i, x] == 0
        else:
            return m.dudx[i, x] == m.v[i, x]
# m.reform_con2 = Constraint(m.i, m.x, rule=_reform_con_2)


# =============================================================================
# Combined mass balances [FIX x==r boundary case]
def _sulfite(m, x):
    # "CaSO3", "HSO3-", "SO32-", "SO2", "MgSO3"
#    if x == 0:
#        return m.u["CaSO3", x] == 0
#    else:
#        expr = 0
#        for k in ["SO2", "HSO3-", "SO32-", "CaSO3", "MgSO3"]:
#            expr += m.Diff[k] * (m.dudx[k, x] - 2 / (x) * m.u[k, x])
#        return expr == 0
#    if x == r:
    if x == 0:
        return m.u["CaSO3", x] == 0
#        return Constraint.Skip
    else:
        expr = 0
        for k in sulfite_species:
            expr += m.Diff[k] * (m.v[k, x] - 2 / (x) * m.u[k, x])
        return expr == 0
# m.sulfite = Constraint(m.x, rule=_sulfite)


def _carbonate(m, x):
#    if x == r:
    if x == 0:
        return m.u["CaHCO3+", x] == 0
#        return Constraint.Skip
    else:
        expr = 0
        for k in carbonate_species:
            expr += m.Diff[k] * (m.v[k, x] - 2 / (x) * m.u[k, x])
        expr += m.rd_CaCO3[x]
        return expr == 0
# m.carbonate = Constraint(m.x, rule=_carbonate)


def _calcium(m, x):
#    if x == r:
    if x == 0:
        return m.u["CaCO3", x] == 0
#        return Constraint.Skip
#        expr = sum((m.c[k, x+dx] - m.c[k, x]) / dx for k in
#                   ["Ca2+", "CaSO3", "CaCO3", "CaHCO3+", "CaSO4"])
#        return expr == 0
#    if x == r:
#        return sum(m.J[k, x] for k in
#                   ["Ca2+", "CaSO3", "CaCO3", "CaHCO3+", "CaSO4"]) == 0
    else:
        expr = 0
        for k in calcium_species:
#        for k in ["Ca2+", "CaSO3", "CaCO3"]:
            expr += m.Diff[k] * (m.v[k, x] - 2 / (x) * m.u[k, x])
        expr += m.rd_CaCO3[x]
        return expr == 0
# m.calcium = Constraint(m.x, rule=_calcium)


def _carbondioxide(m, x):
    k = "CO2"
#    if x == 0:
#        return (m.c[k, x+dx] - m.c[k, x]) / dx == 0
#    if x == r:
    if x == 0:
        return m.u[k, x] == kG[k] * (m.p[k] - m.p_i[k])
#        return Constraint.Skip
#        return m.H_CO2 * m.c[k, x] == m.p[k]
    else:
        expr = m.Diff[k] * (m.v[k, x] - 2 / (x) * m.u[k, x])
        expr += m.r_CO2[x]
        return expr == 0
# m.carbondioxide = Constraint(m.x, rule=_carbondioxide)


def _charge(m, x):
#    if x == r:
    if x == 0:
        return m.u["H+", x] == 0
#        return Constraint.Skip
#        return sum((m.c[k, x+dx] - m.c[k, x]) / dx for k in m.i) == 0
#    if x == r:
#        return sum(m.J[k, x] for k in m.i) == 0
    else:
        expr = sum(m.z[k] * m.Diff[k] * (m.v[k, x] -
                   2 / (x) * m.u[k, x]) for k in charged_species)
        return expr == 0
# m.charge = Constraint(m.x, rule=_charge)


# =============================================================================
def build_droplet_model(m):
    # eqm constraints
    m.water_diss = Constraint(m.x, rule=_water_diss)
    m.H2SO3_diss = Constraint(m.x, rule=_H2SO3_diss)
    m.HSO3_diss = Constraint(m.x, rule=_HSO3_diss)
    m.HCO3_diss = Constraint(m.x, rule=_HCO3_diss)
    m.CaSO3_diss = Constraint(m.x, rule=_CaSO3_diss)
    m.CaCO3_diss = Constraint(m.x, rule=_CaCO3_diss)
    m.CaHCO3_diss = Constraint(m.x, rule=_CaHCO3_diss)
    
    # twice-diff constraints
    m.water_diss_der2 = Constraint(m.x, rule=_water_diss_der2)
    m.H2SO3_diss_der2 = Constraint(m.x, rule=_H2SO3_diss_der2)
    m.HSO3_diss_der2 = Constraint(m.x, rule=_HSO3_diss_der2)
    m.HCO3_diss_der2 = Constraint(m.x, rule=_HCO3_diss_der2)
    m.CaSO3_diss_der2 = Constraint(m.x, rule=_CaSO3_diss_der2)
    m.CaCO3_diss_der2 = Constraint(m.x, rule=_CaCO3_diss_der2)
    m.CaHCO3_diss_der2 = Constraint(m.x, rule=_CaHCO3_diss_der2)

    # mass balance equations
    m.sulfite = Constraint(m.x, rule=_sulfite)
    m.carbonate = Constraint(m.x, rule=_carbonate)
    m.calcium = Constraint(m.x, rule=_calcium)
    m.carbondioxide = Constraint(m.x, rule=_carbondioxide)
    m.charge = Constraint(m.x, rule=_charge)
    
    # reformulation constraints
    m.reform_con = Constraint(m.i, m.x, rule=_reform_con)
    m.reform_con2 = Constraint(m.i, m.x, rule=_reform_con_2)
    
    # objective
    m.objective = Objective(expr=1, sense=minimize)
    
    
    # =========================================================================
    # dicretize
    # discretizer = TransformationFactory("dae.finite_difference")
    # discretizer.apply_to(m, nfe=nfe, wrt=m.x, scheme="BACKWARD")
    discretizer = TransformationFactory("dae.collocation")
    discretizer.apply_to(m, nfe=nfe, ncp=3, wrt=m.x)

    # =========================================================================
    # m.dudx["SO2", 0].fix(0.0)
    # m.dudx["CO2", 0].fix(0.0)
    m.dudx["SO2", r].fix(0.0)
    m.dudx["CO2", r].fix(0.0)
    for comp in diff_vars:
        if comp == "CO2":
            m.dcdx[comp, r].fix(1.0e-06)
        else:
            m.dcdx[comp, r].fix(0.0)


    # =========================================================================
    opt = SolverFactory("ipopt")
    opt.options['linear_solver'] = 'MA27'
    opt.options['bound_push'] = 1e-05
    opt.options['mu_init'] = 1e-05
    opt.options['linear_system_scaling'] = 'mc19'
    #opt.options["nlp_scaling_method"] = "equilibration-based"
    #opt.options["output_file"] = "so2_abs_output.txt"
    opt.options["print_level"] = 5
    #opt.options["file_print_level"] = 7
    opt.options["bound_relax_factor"] = 1e-03
    opt.options["honor_original_bounds"] = "yes"
    #opt.options["max_iter"] = 1


    # sb = TransformationFactory('contrib.strip_var_bounds')
    # sb.apply_to(m,reversible=True)
    results = opt.solve(m, tee=True)
    
    if (results.solver.termination_condition == TerminationCondition.infeasible or
            results.solver.termination_condition ==
            TerminationCondition.maxIterations):
        while opt.options['bound_push'] <= 0.01 and opt.options['mu_init'] <= 0.01:
            opt.options['bound_push'] *= 10
            results = opt.solve(m, tee=True, load_solutions=True)
            if (results.solver.termination_condition ==
                    TerminationCondition.optimal):
                break
            if opt.options['bound_push'] == 0.1 and opt.options['mu_init'] != 0.1:
                opt.options['bound_push'] = 1e-6
                opt.options['mu_init'] *= 10
    
    # sb.revert(m)
    # m.write(filename="so2_abs_reform.nl", format=ProblemFormat.nl)
    
    # results = opt.solve(m, tee=True, load_solutions=True)
    
    
    # =========================================================================
    # Stale vars
    # stale_dudx_vars = [comp for comp in m.dudx if m.dudx[comp[0], comp[1]].stale]
    # stale_dcdx_vars = [comp for comp in m.dcdx if m.dcdx[comp[0], comp[1]].stale]
    
    
    # =========================================================================
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # plt.rc('xtick', labelsize=20) 
    # plt.rc('ytick', labelsize=20)
    # plt.rc('axes', labelsize=25)    # fontsize of the x and y labels
    plt.plot(list(m.x), [value(m.c["SO2", x_]) for x_ in list(m.x)])
    plt.xlabel(r'\Large Radial coordinate [m] (center --$>$ surface)')
    plt.ylabel(r'\Large SO$_2$ Concentration [M]')
    plt.xlim([0.0, r])
    plt.grid()
    plt.tight_layout()
    
    
    plt.figure()
    plt.plot(list(m.x), [value(m.c["CaCO3", x_]) for x_ in list(m.x)])
    plt.xlabel(r'\Large Radial coordinate (center --$>$ surface)')
    plt.ylabel(r'\Large CaCO$_3$ Concentration [M]')
    plt.xlim([0.0, r])
    plt.grid()
    plt.tight_layout()
    
    print("delta C[SO2] =", m.c["SO2", r]() - m.c["SO2", 0]())
    print("C[SO2] at the surface =", m.c["SO2", r]())
    #"""
