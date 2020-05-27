# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:23:55 2020

@author: vdabadgh
"""

from __future__ import division, print_function

# from droplet_model_12_species import m
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

import numpy as np
pi = np.pi
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
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
nfe_r = 50
decimals = 10
dx = round(r / nfe_r, decimals)
r = round(dx * nfe_r, decimals)
x = list(np.round(np.linspace(0, r, num=nfe_r+1), decimals=decimals))
m.x = ContinuousSet(initialize=x)

# Axial coordinate
z = 5  # [m]
nfe_z = 1
dz = round(z / nfe_z, decimals)
z = round(dz * nfe_z, decimals)
z = list(np.round(np.linspace(0, z, num=nfe_z+1), decimals=decimals))
m.z = ContinuousSet(initialize=z)

# Species
spc = [sp for sp in sp_all if sp != "CO2" and sp != "SO2"]
species_removed = ['CaHSO3+', 'MgHSO3+', 'MgSO3', 'MgSO4', 'Mg2+', 'MgHCO3+',
                   'Cl-', 'SO42-', 'CaSO4']
i = [sp for sp in i if sp not in species_removed]
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

# Aggregated species
j = ['sulfite', 'carbonate', 'calcium', 'charged', 'CO2']
m.j = Set(initialize=j)


# =============================================================================
# Params
# =============================================================================
T = 25 + 273.15  # [K]
R = 8.314        # [J/mol-K]
up = 5           # [m/s]

# Equilibrium constants (Radian corp. data, 1970)
Keq_all = {sp: 10**(-vals[0] / T - vals[1] * np.log10(T) - vals[2] * T +
                    vals[3]) for sp, vals in Keq_const.items()}
Keq = {sp: k for sp, k in Keq_all.items() if sp in eq_i and sp != "SO2"}
Keq["SO2"] = Keq_all["H2SO3"]
# Using new K values from other literature:
# Keq['CaSO3'] = 1.35e-05       # Frydman et al. (1958)
# Keq['SO2'] = 7.94e-03         # Ammonia based scrubber paper (2011)
# Keq['HSO3-'] = 4.4e-08        # Ammonia based scrubber paper (2011)
# Keq['HCO3-'] = 8.2e-10        # Roy et al. (1993)
# Keq['CaCO3'] = 10**(-3.2)     # Jacobson and Langmuir (1973)
# Keq['CaHCO3+'] = 10**(-1.28)  # Jacobson and Langmuir (1973)
m.Keq = Param(m.eq_i, initialize=Keq)
Kw = Keq_all["H2O"]
m.Kw = Param(initialize=Kw)

# Diffusivity [m**2/s]
m.Diff = Param(m.i, rule=(lambda m, i: Diff[i]))

# charge
z_sp["CO2"], z_sp["SO2"] = 0, 0
m.q = Param(m.i, rule=(lambda m, i: z_sp[i]))

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
m.c = Var(m.i, m.x, m.z, initialize=1e-03)
m.u = Var(m.i, m.x, m.z, initialize=1e-06)
m.v = Var(m.i, m.x, m.z, initialize=1e-03)
m.dcdx = DerivativeVar(m.c, wrt=(m.x), initialize=1e-03)
m.dudx = DerivativeVar(m.u, wrt=(m.x), initialize=0.0)

m.ctot = Var(m.j, m.x, m.z, initialize=1e-03)
m.dctotdz = DerivativeVar(m.ctot, wrt=(m.z), initialize=1e-03)


# =============================================================================
# Model
# =============================================================================
# Relative supersaturation
def _RS_CaCO3(m, x, z):
    return m.c["Ca2+", x, z] * m.c["CO32-", x, z] / Keq_all["CaCO3(s)"]


m.RS_CaCO3 = Expression(m.x, m.z, rule=_RS_CaCO3)


def _RS_CaSO3(m, x, z):
    return m.c["Ca2+", x, z] * m.c["SO32-", x, z] / Keq_all["CaSO3(s)"]


m.RS_CaSO3 = Expression(m.x, m.z, rule=_RS_CaSO3)


# =============================================================================
# Finite rate reactions occuring in the slurry

# Hydrolysis of CO2
# Forward rate of reaction (Pinsent et al. 1956) [1/s]
logk1_CO2 = 329.85 - 110.541 * np.log10(T) - 17265.5 / T
k1_CO2 = 10**(logk1_CO2)


def _r_CO2(m, x, z):
    keq_CO2 = Keq_all["H2CO3"]
    return k1_CO2 * m.c["CO2", x, z] * (m.c["H+", x, z] * m.c["HCO3-", x, z] /
                                     (keq_CO2 * m.c["CO2", x, z]) - 1)


m.r_CO2 = Expression(m.x, m.z, rule=_r_CO2)


# =============================================================================
# Limestone dissolution
def _rd_CaCO3(m, x, z):
#    if x == r:  # 0 to skip at surface
    if x == 0:
        return Expression.Skip
    else:
        xi = 0
        for k in ["HCO3-", "CO32-", "CaCO3"]:
            xi += -m.Diff[k] * m.u[k, x, z]
        return 6 * m_limestone / (d * rho_l) * xi


m.rd_CaCO3 = Expression(m.x, m.z, rule=_rd_CaCO3)


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
# Aggregated concentrations
def _ctot_sulfite(m, x, z):
    return m.ctot['sulfite', x, z] == sum(m.c[k, x, z] for k in sulfite_species)
m.ctot_sulfite = Constraint(m.x, m.z, rule=_ctot_sulfite)


def _ctot_carbonate(m, x, z):
    return m.ctot['carbonate', x, z] == sum(m.c[k, x, z] for k in carbonate_species)
m.ctot_carbonate = Constraint(m.x, m.z, rule=_ctot_carbonate)


def _ctot_calcium(m, x, z):
    return m.ctot['calcium', x, z] == sum(m.c[k, x, z] for k in calcium_species)
m.ctot_calcium = Constraint(m.x, m.z, rule=_ctot_calcium)


def _ctot_charged(m, x, z):
    return m.ctot['charged', x, z] == sum(m.c[k, x, z] for k in charged_species)
m.ctot_charged = Constraint(m.x, m.z, rule=_ctot_charged)


def _ctot_co2(m, x, z):
    return m.ctot['CO2', x, z] == m.c['CO2', x, z]
m.ctot_co2 = Constraint(m.x, m.z, rule=_ctot_co2)


# =============================================================================
# Equilibrium reactions
def _water_diss(m, x, z):
    return m.c["H+", x, z] * m.c["OH-", x, z] == m.Kw
m.water_diss = Constraint(m.x, m.z, rule=_water_diss)


def _H2SO3_diss(m, x, z):  # use Keq[H2SO3]
    return m.Keq["SO2"] * m.c["SO2", x, z] == m.c["H+", x, z] * m.c["HSO3-", x, z]
m.H2SO3_diss = Constraint(m.x, m.z, rule=_H2SO3_diss)


def _HSO3_diss(m, x, z):
    return m.Keq["HSO3-"] * m.c["HSO3-", x, z] == m.c["H+", x, z] * m.c["SO32-", x, z]
m.HSO3_diss = Constraint(m.x, m.z, rule=_HSO3_diss)


def _HCO3_diss(m, x, z):
    return m.Keq["HCO3-"] * m.c["HCO3-", x, z] == m.c["H+", x, z] * m.c["CO32-", x, z]
m.HCO3_diss = Constraint(m.x, m.z, rule=_HCO3_diss)


def _CaSO3_diss(m, x, z):
    return m.Keq["CaSO3"] * m.c["CaSO3", x, z] == m.c["Ca2+", x, z] * m.c["SO32-", x, z]
m.CaSO3_diss = Constraint(m.x, m.z, rule=_CaSO3_diss)


def _CaCO3_diss(m, x, z):
    return m.Keq["CaCO3"] * m.c["CaCO3", x, z] == m.c["Ca2+", x, z] * m.c["CO32-", x, z]
m.CaCO3_diss = Constraint(m.x, m.z, rule=_CaCO3_diss)


def _CaHCO3_diss(m, x, z):
    return m.Keq["CaHCO3+"] * m.c["CaHCO3+", x, z] == m.c["Ca2+", x, z] * m.c["HCO3-", x, z]
m.CaHCO3_diss = Constraint(m.x, m.z, rule=_CaHCO3_diss)


# =============================================================================
# Twice-differentiated equations
def _water_diss_der2(m, x, z):
    return m.v["H+", x, z] * m.c["OH-", x, z] + \
        m.c["H+", x, z] * m.v["OH-", x, z] + \
        2 * m.u["H+", x, z] * m.u["OH-", x, z] == 0
m.water_diss_der2 = Constraint(m.x, m.z, rule=_water_diss_der2)


def _H2SO3_diss_der2(m, x, z):  # use Keq[H2SO3]
    return m.Keq["SO2"] * m.v["SO2", x, z] == \
        m.v["H+", x, z] * m.c["HSO3-", x, z] + \
        m.c["H+", x, z] * m.v["HSO3-", x, z] + \
        2 * m.u["H+", x, z] * m.u["HSO3-", x, z]
m.H2SO3_diss_der2 = Constraint(m.x, m.z, rule=_H2SO3_diss_der2)


def _HSO3_diss_der2(m, x, z):
    return m.Keq["HSO3-"] * m.v["HSO3-", x, z] == \
        m.v["H+", x, z] * m.c["SO32-", x, z] + \
        m.c["H+", x, z] * m.v["SO32-", x, z] + \
        2 * m.c["H+", x, z] * m.u["SO32-", x, z]
m.HSO3_diss_der2 = Constraint(m.x, m.z, rule=_HSO3_diss_der2)


def _HCO3_diss_der2(m, x, z):
    return m.Keq["HCO3-"] * m.v["HCO3-", x, z] == \
        m.v["H+", x, z] * m.c["CO32-", x, z] + \
        m.c["H+", x, z] * m.v["CO32-", x, z] + \
        2 * m.u["H+", x, z] * m.u["CO32-", x, z]
m.HCO3_diss_der2 = Constraint(m.x, m.z, rule=_HCO3_diss_der2)


def _CaSO3_diss_der2(m, x, z):
    return m.Keq["CaSO3"] * m.v["CaSO3", x, z] == \
        m.v["Ca2+", x, z] * m.c["SO32-", x, z] + \
        m.c["Ca2+", x, z] * m.v["SO32-", x, z] + \
        2 * m.u["Ca2+", x, z] * m.u["SO32-", x, z]
m.CaSO3_diss_der2 = Constraint(m.x, m.z, rule=_CaSO3_diss_der2)


def _CaCO3_diss_der2(m, x, z):
    return m.Keq["CaCO3"] * m.v["CaCO3", x, z] == \
        m.v["Ca2+", x, z] * m.c["CO32-", x, z] + \
        m.c["Ca2+", x, z] * m.v["CO32-", x, z] + \
        2 * m.u["Ca2+", x, z] * m.u["CO32-", x, z]
m.CaCO3_diss_der2 = Constraint(m.x, m.z, rule=_CaCO3_diss_der2)


def _CaHCO3_diss_der2(m, x, z):
    return m.Keq["CaHCO3+"] * m.v["CaHCO3+", x, z] == \
        m.v["Ca2+", x, z] * m.c["HCO3-", x, z] + \
        m.c["Ca2+", x, z] * m.v["HCO3-", x, z] + \
        2 * m.u["Ca2+", x, z] * m.u["HCO3-", x, z]
m.CaHCO3_diss_der2 = Constraint(m.x, m.z, rule=_CaHCO3_diss_der2)


# =============================================================================
# Define u = dcdx
def _reform_con(m, i, x, z):
    if i in diff_vars:
        if x == r:
            return m.c[i, x, z] == 0.0005
        else:
            return m.u[i, x, z] == m.dcdx[i, x, z]
    else:
        return Constraint.Skip
m.reform_con = Constraint(m.i, m.x, m.z, rule=_reform_con)


# =============================================================================
# Define dudx = v
def _reform_con_2(m, i, x, z):
    if i == "SO2" or i == "CO2":
        if x == r:
            return m.u[i, x, z] == kG[i] * (p[i] - p_i[i])
        else:
            return m.dudx[i, x, z] == m.v[i, x, z]
    else:
        if x == 0:
            return m.u[i, x, z] == 0
        else:
            return m.dudx[i, x, z] == m.v[i, x, z]
m.reform_con2 = Constraint(m.i, m.x, m.z, rule=_reform_con_2)


# =============================================================================
# Combined mass balances [FIX x==r boundary case]
def _sulfite(m, x, z):
    if z == 0:
        return m.ctot['sulfite', x, z] == 5e-04
    else:
        if x == 0:
            return m.u["CaSO3", x, z] == 0
        else:
            expr = 0
            for k in sulfite_species:
                expr += m.Diff[k] * (m.v[k, x, z] - 2 / (x) * m.u[k, x, z])
            return expr == up * m.dctotdz['sulfite', x, z]
m.sulfite = Constraint(m.x, m.z, rule=_sulfite)


def _carbonate(m, x, z):
    if z == 0:
        return m.ctot['carbonate', x, z] == 2e-03
    else:
        if x == 0:
            return m.u["CaHCO3+", x, z] == 0
        else:
            expr = 0
            for k in carbonate_species:
                expr += m.Diff[k] * (m.v[k, x, z] - 2 / (x) * m.u[k, x, z])
            expr += m.rd_CaCO3[x, z]
            return expr == up * m.dctotdz['carbonate', x, z]
m.carbonate = Constraint(m.x, m.z, rule=_carbonate)


def _calcium(m, x, z):
    if z == 0:
        return m.ctot['calcium', x, z] == 0.12
    else:
        if x == 0:
            return m.u["CaCO3", x, z] == 0
        else:
            expr = 0
            for k in calcium_species:
                expr += m.Diff[k] * (m.v[k, x, z] - 2 / (x) * m.u[k, x, z])
            expr += m.rd_CaCO3[x, z]
            return expr == up * m.dctotdz['calcium', x, z]
m.calcium = Constraint(m.x, m.z, rule=_calcium)


def _carbondioxide(m, x, z):
    k = 'CO2'
    if z == 0:
        return m.c[k, x, z] == 2e-03
    else:
        if x == 0:
            return m.u[k, x, z] == kG[k] * (m.p[k] - m.p_i[k])
        else:
            expr = m.Diff[k] * (m.v[k, x, z] - 2 / (x) * m.u[k, x, z])
            expr += m.r_CO2[x, z]
            return expr == up * m.dctotdz[k, x, z]
m.carbondioxide = Constraint(m.x, m.z, rule=_carbondioxide)


def _charge(m, x, z):
    if z == 0:
        return m.ctot['charged', x, z] == 1e-01
    else:
        if x == 0:
            return m.u['H+', x, z] == 0
        else:
            expr = sum(m.q[k] * m.Diff[k] * (m.v[k, x, z] -
                       2 / (x) * m.u[k, x, z]) for k in charged_species)
            return expr == up * m.dctotdz['charged', x, z]
m.charge = Constraint(m.x, m.z, rule=_charge)


# =============================================================================
# def build_droplet_model(m):
#     # eqm constraints
#     m.water_diss = Constraint(m.x, rule=_water_diss)
#     m.H2SO3_diss = Constraint(m.x, rule=_H2SO3_diss)
#     m.HSO3_diss = Constraint(m.x, rule=_HSO3_diss)
#     m.HCO3_diss = Constraint(m.x, rule=_HCO3_diss)
#     m.CaSO3_diss = Constraint(m.x, rule=_CaSO3_diss)
#     m.CaCO3_diss = Constraint(m.x, rule=_CaCO3_diss)
#     m.CaHCO3_diss = Constraint(m.x, rule=_CaHCO3_diss)
    
#     # twice-diff constraints
#     m.water_diss_der2 = Constraint(m.x, rule=_water_diss_der2)
#     m.H2SO3_diss_der2 = Constraint(m.x, rule=_H2SO3_diss_der2)
#     m.HSO3_diss_der2 = Constraint(m.x, rule=_HSO3_diss_der2)
#     m.HCO3_diss_der2 = Constraint(m.x, rule=_HCO3_diss_der2)
#     m.CaSO3_diss_der2 = Constraint(m.x, rule=_CaSO3_diss_der2)
#     m.CaCO3_diss_der2 = Constraint(m.x, rule=_CaCO3_diss_der2)
#     m.CaHCO3_diss_der2 = Constraint(m.x, rule=_CaHCO3_diss_der2)

#     # mass balance equations
#     m.sulfite = Constraint(m.x, rule=_sulfite)
#     m.carbonate = Constraint(m.x, rule=_carbonate)
#     m.calcium = Constraint(m.x, rule=_calcium)
#     m.carbondioxide = Constraint(m.x, rule=_carbondioxide)
#     m.charge = Constraint(m.x, rule=_charge)
    
#     # reformulation constraints
#     m.reform_con = Constraint(m.i, m.x, rule=_reform_con)
#     m.reform_con2 = Constraint(m.i, m.x, rule=_reform_con_2)
    

# =============================================================================
# objective
m.objective = Objective(expr=1, sense=minimize)


# =============================================================================
# dicretize
discretizer1 = TransformationFactory("dae.collocation")
discretizer1.apply_to(m, nfe=nfe_r, ncp=3, wrt=m.x)
discretizer2 = TransformationFactory("dae.finite_difference")
discretizer2.apply_to(m, nfe=nfe_z, wrt=m.z, scheme="BACKWARD")
# discretizer2 = TransformationFactory("dae.collocation")
# discretizer2.apply_to(m, nfe=nfe_z, ncp=3, wrt=m.z)


# =============================================================================
# m.dudx["SO2", 0].fix(0.0)
# m.dudx["CO2", 0].fix(0.0)
# for z_ in m.z:
#     m.dudx["SO2", r, z_].fix(0.0)
#     m.dudx["CO2", r, z_].fix(0.0)
# for comp in diff_vars:
#     for z_ in m.z:
#         if comp == "CO2":
#             m.dcdx[comp, r, z_].fix(1.0e-06)
#         else:
#             m.dcdx[comp, r, z_].fix(0.0)


# =============================================================================
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


# =============================================================================
# Stale vars
# stale_dudx_vars = [comp for comp in m.dudx if m.dudx[comp[0], comp[1]].stale]
# stale_dcdx_vars = [comp for comp in m.dcdx if m.dcdx[comp[0], comp[1]].stale]
    
    
# =============================================================================
# plt.figure()
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# # plt.rc('xtick', labelsize=20) 
# # plt.rc('ytick', labelsize=20)
# # plt.rc('axes', labelsize=25)    # fontsize of the x and y labels
# plt.plot(list(m.x), [value(m.c["SO2", x_]) for x_ in list(m.x)])
# plt.xlabel(r'\Large Radial coordinate [m] (center --$>$ surface)')
# plt.ylabel(r'\Large SO$_2$ Concentration [M]')
# plt.xlim([0.0, r])
# plt.grid()
# plt.tight_layout()


# plt.figure()
# plt.plot(list(m.x), [value(m.c["CaCO3", x_]) for x_ in list(m.x)])
# plt.xlabel(r'\Large Radial coordinate (center --$>$ surface)')
# plt.ylabel(r'\Large CaCO$_3$ Concentration [M]')
# plt.xlim([0.0, r])
# plt.grid()
# plt.tight_layout()

# print("delta C[SO2] =", m.c["SO2", r]() - m.c["SO2", 0]())
# print("C[SO2] at the surface =", m.c["SO2", r]())
#"""
