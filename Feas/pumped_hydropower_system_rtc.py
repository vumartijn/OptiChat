from pyomo.environ import *
import random

SEED = 42
random.seed(SEED)

# Create a model
model = ConcreteModel(name="PumpedHydropowerSystemRTC")

# =========================================================================
# Sets
# =========================================================================
n_steps = 21
dt = 3600.0  # [s] time step (hourly)

model.T = RangeSet(0, n_steps - 1)
model.T_interior = RangeSet(1, n_steps - 1)

# =========================================================================
# Parameters — physical constants
# =========================================================================
efficiency_reservoir = 0.88   # [-] lower-reservoir turbine efficiency
efficiency_pump      = 0.70   # [-] pump efficiency
efficiency_turbine   = 0.90   # [-] upper-reservoir turbine efficiency
gravity = 9.81                # [m/s²] gravitational acceleration
rho     = 1000.0              # [kg/m³] water density
fix_dH_reservoir = 2.5        # [m] fixed head over the lower-reservoir turbine
fix_dH_pump      = 10.0       # [m] fixed head over the pump
fix_dH_turbine   = 10.0       # [m] fixed head over the upper-reservoir turbine

M = 200.0  # [-] big-M constant for the pump/turbine exclusivity logic

# Power conversion coefficients: Power = coefficient * VolumeFlowRate
c_pump      = efficiency_pump      * fix_dH_pump      * gravity * rho  # [J/m³]
c_turbine   = efficiency_turbine   * fix_dH_turbine   * gravity * rho  # [J/m³]
c_reservoir = efficiency_reservoir * fix_dH_reservoir * gravity * rho  # [J/m³]

# Constant inflow from the hinterland into the lower basin [m³/s]
Inflow_Q = 100.0

# --- Randomised scenario parameters (vary SEED to generate new scenarios) ---
# Initial lower-basin volume [m³] and lower-reservoir turbine capacity [m³/s].
# Shrinking the turbine capacity can make the high power targets infeasible.
V_LB_initial = 400000.0 + random.uniform(0.0, 200000.0)
V_UB_initial = 0.0
ReservoirTurbineFlow_max = 100.0 - random.uniform(0.0, 40.0)

# =========================================================================
# Parameters — time series
# =========================================================================
# Target electrical power the system must deliver at each time step [W]
Target_Power_data = {
    0: 1.0e6,  1: 1.0e6,  2: 1.0e6,  3: 1.0e6,  4: 1.0e6,
    5: 1.0e6,  6: 2.5e6,  7: 2.5e6,  8: 2.5e6,  9: 2.5e6,
    10: 2.5e6, 11: 1.0e6, 12: 1.0e6, 13: 1.0e6, 14: 1.0e6,
    15: 1.0e6, 16: 2.5e6, 17: 2.5e6, 18: 2.5e6, 19: 2.5e6,
    20: 2.5e6,
}

# Electricity price signal [$/W]
cost_perP_data = {
    0: 0.019,  1: 0.019,  2: 0.019,  3: 0.013,  4: 0.013,
    5: 0.013,  6: 0.019,  7: 0.019,  8: 0.019,  9: 0.019,
    10: 0.019, 11: 0.013, 12: 0.013, 13: 0.013, 14: 0.013,
    15: 0.013, 16: 0.019, 17: 0.019, 18: 0.019, 19: 0.019,
    20: 0.019,
}

model.Target_Power = Param(model.T, initialize=Target_Power_data, mutable=True,
                           doc='Target electrical power output at each time step [W]')
model.cost_perP    = Param(model.T, initialize=cost_perP_data, mutable=True,
                           doc='Electricity price signal at each time step [$/W]')

# =========================================================================
# Variables
# =========================================================================
model.V_LowerBasin = Var(model.T, bounds=(0.0, 1.0e7), initialize=V_LB_initial,
                         doc='Lower reservoir water volume [m³]')
model.V_UpperBasin = Var(model.T, bounds=(0.0, 1.0e7), initialize=V_UB_initial,
                         doc='Upper basin water volume [m³]')
model.PumpFlow     = Var(model.T, bounds=(0.0, 10.0), initialize=0.0,
                         doc='Flow pumped from the lower reservoir up to the upper basin [m³/s]')
model.TurbineFlow  = Var(model.T, bounds=(0.0, 10.0), initialize=0.0,
                         doc='Flow released from the upper basin through the turbine [m³/s]')
model.ReservoirTurbineFlow = Var(model.T, bounds=(0.0, ReservoirTurbineFlow_max), initialize=0.0,
                                 doc='Flow through the lower-reservoir turbine [m³/s]')
model.ReservoirSpillFlow   = Var(model.T, bounds=(0.0, 100.0), initialize=0.0,
                                 doc='Flow spilled from the lower reservoir (bypasses the turbine) [m³/s]')
model.Turbine_is_on = Var(model.T_interior, domain=Binary, initialize=0,
                          doc='Binary indicator: 1 if the reversible unit acts as a turbine, 0 if as a pump')

# =========================================================================
# Expressions — power and revenue (algebraic outputs)
# =========================================================================
model.PumpPower      = Expression(model.T, rule=lambda m, t: c_pump * m.PumpFlow[t],
                                  doc='Electrical power consumed by the pump [W]')
model.TurbinePower   = Expression(model.T, rule=lambda m, t: c_turbine * m.TurbineFlow[t],
                                  doc='Electrical power generated by the upper-reservoir turbine [W]')
model.ReservoirPower = Expression(model.T, rule=lambda m, t: c_reservoir * m.ReservoirTurbineFlow[t],
                                  doc='Electrical power generated by the lower-reservoir turbine [W]')
model.TotalGeneratingPower = Expression(
    model.T, rule=lambda m, t: m.ReservoirPower[t] + m.TurbinePower[t],
    doc='Total power generated by both turbines [W]')
model.TotalSystemPower = Expression(
    model.T, rule=lambda m, t: m.ReservoirPower[t] + m.TurbinePower[t] - m.PumpPower[t],
    doc='Net power delivered by the system (generation minus pumping) [W]')
model.PumpCost = Expression(
    model.T, rule=lambda m, t: m.PumpPower[t] * m.cost_perP[t],
    doc='Cost of running the pump [$]')
model.SystemGeneratingRevenue = Expression(
    model.T, rule=lambda m, t: m.TotalGeneratingPower[t] * m.cost_perP[t],
    doc='Revenue from generated power [$]')
model.TotalSystemRevenue = Expression(
    model.T, rule=lambda m, t: m.SystemGeneratingRevenue[t] - m.PumpCost[t],
    doc='Net revenue of the system (generating revenue minus pump cost) [$]')

# =========================================================================
# Constraints
# =========================================================================

# Initial state (t=0) is fixed to the measured initial condition; at the
# start no flows are active and the basin volumes are known.
model.V_LowerBasin[0].fix(V_LB_initial)
model.V_UpperBasin[0].fix(V_UB_initial)
model.PumpFlow[0].fix(0.0)
model.TurbineFlow[0].fix(0.0)
model.ReservoirTurbineFlow[0].fix(0.0)
model.ReservoirSpillFlow[0].fix(0.0)

# Mass balance of the lower reservoir: inflow plus the pump and turbine
# lateral flows, minus the reservoir turbine and spill releases.
def lower_mass_balance_rule(model, t):
    return (
        model.V_LowerBasin[t] - model.V_LowerBasin[t - 1]
        == dt * (
            Inflow_Q
            - model.ReservoirTurbineFlow[t]
            - model.ReservoirSpillFlow[t]
            + model.PumpFlow[t]
            + model.TurbineFlow[t]
        )
    )
model.lower_mass_balance = Constraint(model.T_interior, rule=lower_mass_balance_rule,
                                      doc='Water volume continuity for the lower reservoir')

# Mass balance of the upper basin: filled by the pump, drained by the turbine.
def upper_mass_balance_rule(model, t):
    return (
        model.V_UpperBasin[t] - model.V_UpperBasin[t - 1]
        == dt * (model.PumpFlow[t] - model.TurbineFlow[t])
    )
model.upper_mass_balance = Constraint(model.T_interior, rule=upper_mass_balance_rule,
                                      doc='Water volume continuity for the upper basin')

# Pump/turbine exclusivity (big-M). The reversible unit cannot pump and
# generate at the same time. Turbine_is_on=1 forces PumpFlow=0, while
# Turbine_is_on=0 forces TurbineFlow=0. The first two constraints are the
# trivially-satisfied counterparts kept to mirror the RTC-Tools formulation.
def bigM_pump_lower_rule(model, t):
    return 0.0 <= model.PumpFlow[t] + model.Turbine_is_on[t] * M
model.bigM_pump_lower = Constraint(model.T_interior, rule=bigM_pump_lower_rule,
                                   doc='Big-M lower bound on pump flow')

def bigM_turbine_lower_rule(model, t):
    return 0.0 <= model.TurbineFlow[t] + (1 - model.Turbine_is_on[t]) * M
model.bigM_turbine_lower = Constraint(model.T_interior, rule=bigM_turbine_lower_rule,
                                      doc='Big-M lower bound on turbine flow')

def bigM_pump_off_rule(model, t):
    return model.PumpFlow[t] - (1 - model.Turbine_is_on[t]) * M <= 0.0
model.bigM_pump_off = Constraint(model.T_interior, rule=bigM_pump_off_rule,
                                 doc='Forces pump flow to zero while the unit generates')

def bigM_turbine_off_rule(model, t):
    return model.TurbineFlow[t] - model.Turbine_is_on[t] * M <= 0.0
model.bigM_turbine_off = Constraint(model.T_interior, rule=bigM_turbine_off_rule,
                                    doc='Forces turbine flow to zero while the unit pumps')

# Goal (priority 10): the net system power must meet the requested target.
def power_target_rule(model, t):
    return model.TotalSystemPower[t] == model.Target_Power[t]
model.power_target = Constraint(model.T_interior, rule=power_target_rule,
                                doc='Net system power must equal the target power')

# Goal (priority 20): no water may be spilled past the lower-reservoir turbine.
def no_spill_rule(model, t):
    return model.ReservoirSpillFlow[t] == 0.0
model.no_spill = Constraint(model.T_interior, rule=no_spill_rule,
                            doc='Forbid spill flow so all release passes the turbine')

# =========================================================================
# Objective
# =========================================================================
# Merges the two lowest-priority goals of the RTC-Tools problem: minimising
# the pump cost (priority 25) and maximising the generating revenue
# (priority 30). Both are captured by maximising the net system revenue.
model.obj = Objective(
    expr=sum(model.TotalSystemRevenue[t] for t in model.T),
    sense=maximize,
    doc='Maximise total net revenue over the planning horizon [$]')
