import math
from pyomo.environ import *


# Create a model
model = ConcreteModel(name="LP_RTC")

# =========================================================================
# Sets
# =========================================================================
n_steps = 21
dt = 3600.0  # [s] time step

model.T = RangeSet(0, n_steps - 1)
model.T_interior = RangeSet(1, n_steps - 1)

# =========================================================================
# Parameters — physical constants
# =========================================================================
A = 1.0e6    # [m²] storage area
M = 2.0      # [-]  big-M constant for logical constraints
w = 3.0      # [m]  orifice width
d = 0.8      # [m]  orifice height
C = 1.0      # [-]  orifice discharge coefficient
g = 9.8      # [m/s²] gravitational acceleration
Q_orifice_max = 10.0  # [m³/s]

# Piecewise linear approximation of Torricelli: Q(dH) = w*C*d * sqrt(2*g*dH)
# Three segments defined by breakpoints dH in [0,0.01], [0.01,0.1], [0.1,0.4]
_dH = [0.0, 0.01, 0.1, 0.4]
_Q  = [w * C * d * math.sqrt(2 * g * h) if h > 0 else 0.0 for h in _dH]
# _Q  ≈ [0.0, 1.0625, 3.36, 6.72]

slope1     = (_Q[1] - _Q[0]) / (_dH[1] - _dH[0])   # ≈ 106.25
slope2     = (_Q[2] - _Q[1]) / (_dH[2] - _dH[1])   # ≈ 25.53
slope3     = (_Q[3] - _Q[2]) / (_dH[3] - _dH[2])   # = 11.2
intercept1 = _Q[0] - slope1 * _dH[0]                # = 0.0
intercept2 = _Q[1] - slope2 * _dH[1]                # ≈ 0.807
intercept3 = _Q[2] - slope3 * _dH[2]                # = 2.24

# Tidal sea level [m]
H_sea_data = {
    0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3,  4: 0.4,
    5: 0.5, 6: 0.6, 7: 0.7, 8: 0.8,  9: 0.9,
    10: 1.0, 11: 0.9, 12: 0.8, 13: 0.7, 14: 0.6,
    15: 0.5, 16: 0.4, 17: 0.3, 18: 0.2, 19: 0.1,
    20: 0.0,
}

# Constant inflow [m³/s]
Q_in_data = {t: 5.0 for t in range(n_steps)}

model.H_sea = Param(model.T, initialize=H_sea_data, mutable=True,
                    doc='Sea water level at each time step [m]')
model.Q_in  = Param(model.T, initialize=Q_in_data,  mutable=True,
                    doc='Inflow discharge from hinterland at each time step [m³/s]')
model.H_initial     = Param(initialize=0.4, mutable=True,
                            doc='Initial storage level [m]')
model.H_storage_max = Param(initialize=0.5, mutable=True,
                            doc='Maximum allowed storage level (basin capacity limit) [m]')
model.Q_pump_max    = Param(initialize=7.0, mutable=True,
                            doc='Maximum pump discharge rate [m³/s]')

# =========================================================================
# Variables
# =========================================================================
model.H_storage  = Var(model.T, bounds=(0.0, None), initialize=value(model.H_initial),
                       doc='Storage water level [m]')
model.Q_pump     = Var(model.T, bounds=(0.0, None), initialize=0.0,
                       doc='Pump discharge rate [m³/s]')
model.Q_orifice  = Var(model.T, bounds=(0.0, Q_orifice_max), initialize=0.0,
                       doc='Orifice (gravity) discharge rate [m³/s]')
# LP relaxation: is_downhill is continuous in [0, 1] instead of binary
model.is_downhill = Var(model.T, bounds=(0.0, 1.0), initialize=0.0,
                        doc='Relaxed indicator: 1 if storage level exceeds sea level (gravity flow possible)')

# =========================================================================
# Constraints
# =========================================================================

# Initial water level
model.initial_condition = Constraint(
    expr=model.H_storage[0] == model.H_initial,
    doc='Fix storage level at t=0 to initial state')

# Basin capacity: enforce H_storage_max as a mutable upper bound via constraint
def storage_upper_bound_rule(model, t):
    return model.H_storage[t] <= model.H_storage_max
model.storage_upper_bound = Constraint(model.T, rule=storage_upper_bound_rule,
                                       doc='Storage level cannot exceed basin capacity limit [m]')

# Mass balance: A*(H[t] - H[t-1]) = dt*(Q_in[t-1] - Q_pump[t-1] - Q_orifice[t-1])
def mass_balance_rule(model, t):
    return (
        A * (model.H_storage[t] - model.H_storage[t - 1])
        == dt * (model.Q_in[t - 1] - model.Q_pump[t - 1] - model.Q_orifice[t - 1])
    )
model.mass_balance = Constraint(model.T_interior, rule=mass_balance_rule,
                                doc='Water volume continuity equation (forward Euler)')

# Pump capacity: enforce Q_pump_max as a mutable upper bound via constraint
def pump_capacity_rule(model, t):
    return model.Q_pump[t] <= model.Q_pump_max
model.pump_capacity = Constraint(model.T, rule=pump_capacity_rule,
                                 doc='Pump discharge cannot exceed maximum pump capacity [m³/s]')

# Orifice can only flow when is_downhill=1
def orifice_downhill_only_rule(model, t):
    return model.Q_orifice[t] <= Q_orifice_max * model.is_downhill[t]
model.orifice_downhill_only = Constraint(model.T, rule=orifice_downhill_only_rule,
                                         doc='Orifice flow is zero when storage level is below sea level')

# is_downhill=1 is only allowed when H_storage >= H_sea
def is_downhill_upper_rule(model, t):
    return model.H_sea[t] - model.H_storage[t] <= M * (1 - model.is_downhill[t])
model.is_downhill_upper = Constraint(model.T, rule=is_downhill_upper_rule,
                                     doc='Forces is_downhill=0 when sea level exceeds storage level (big-M)')

# is_downhill=0 is forced when H_sea > H_storage
def is_downhill_lower_rule(model, t):
    return model.H_sea[t] - model.H_storage[t] + M * model.is_downhill[t] >= 0
model.is_downhill_lower = Constraint(model.T, rule=is_downhill_lower_rule,
                                     doc='Prevents is_downhill=1 when storage is above sea level (big-M)')

# Piecewise linear orifice hydraulic capacity (3 segments, linearised Torricelli).
# All three constraints are active simultaneously; their pointwise minimum forms
# the piecewise linear upper bound over dH = H_storage - H_sea in [0, 0.4].
# The big-M term relaxes each constraint to Q <= Q_orifice_max when is_downhill=0.

def orifice_capacity_seg1_rule(model, t):
    # Active range: H_storage - H_sea in [0, 0.01]
    return (model.Q_orifice[t]
            <= slope1 * (model.H_storage[t] - model.H_sea[t])
               + intercept1 + M * (1 - model.is_downhill[t]))
model.orifice_capacity_seg1 = Constraint(model.T, rule=orifice_capacity_seg1_rule,
                                         doc='Orifice linear capacity: segment dH in [0, 0.01]')

def orifice_capacity_seg2_rule(model, t):
    # Active range: H_storage - H_sea in [0.01, 0.1]
    return (model.Q_orifice[t]
            <= slope2 * (model.H_storage[t] - model.H_sea[t])
               + intercept2 + M * (1 - model.is_downhill[t]))
model.orifice_capacity_seg2 = Constraint(model.T, rule=orifice_capacity_seg2_rule,
                                         doc='Orifice linear capacity: segment dH in [0.01, 0.1]')

def orifice_capacity_seg3_rule(model, t):
    # Active range: H_storage - H_sea in [0.1, 0.4]
    return (model.Q_orifice[t]
            <= slope3 * (model.H_storage[t] - model.H_sea[t])
               + intercept3 + M * (1 - model.is_downhill[t]))
model.orifice_capacity_seg3 = Constraint(model.T, rule=orifice_capacity_seg3_rule,
                                         doc='Orifice linear capacity: segment dH in [0.1, 0.4]')

# =========================================================================
# Objective
# =========================================================================
model.obj = Objective(
    expr=sum(model.Q_pump[t] * dt for t in model.T),
    sense=minimize,
    doc='Minimize total volume of water pumped over the planning horizon [m³]')
