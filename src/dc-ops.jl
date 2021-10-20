
#### DC Optimal Power Shutoff ####

# This file provides a basic implentation of the DC Optimal Power Shutoff
# problem for experimentation on algorithm and problem variations.

# This file can be run by calling `include("dc-ops.jl")` from the Julia REPL

# Developed by Noah Rhodes(@noahrhodes) and Line Roald (@lroald)


###############################################################################
# 0. Initialization
###############################################################################

# Load Julia Packages
#--------------------
using PowerModels
using PowerModelsWildfire
using GLPK
using JuMP


# Load System Data
# ----------------
powermodels_path = joinpath(dirname(pathof(PowerModelsWildfire)), "..")

# datasets with risk data, available in the PowerModelsWildfire.jl package
# file_name = "$(powermodels_path)/test/data/matpower/RTS_GMLC_risk.m"
file_name = "$(powermodels_path)/test/networks/case14_risk.m"
# file_name = "$(powermodels_path)/test/data/matpower/case5_risk_sys1.m"
# file_name = "$(powermodels_path)/test/data/matpower/case5_risk_sys2.m "

# load the data file
data = PowerModels.parse_file(file_name)

# Add zeros to turn linear objective functions into quadratic ones
# so that additional parameter checks are not required
PowerModels.standardize_cost_terms!(data, order=2)

# Adds reasonable rate_a values to branches without them
PowerModels.calc_thermal_limits!(data)

# use build_ref to filter out inactive components
ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
# Note: ref contains all the relevant system parameters needed to build the OPS model
# When we introduce constraints and variable bounds below, we use the parameters in ref.


###############################################################################
# 1. Building the Optimal Power Shutoff Model
###############################################################################

# Initialize a JuMP Optimization Model
#-------------------------------------
model = Model(GLPK.Optimizer)

# set_optimizer_attribute(model, "print_level", 0)
# note: print_level changes the amount of solver information printed to the terminal


# Add Optimization and State Variables
# ------------------------------------

# Add voltage angles va for each bus
@variable(model, va[i in keys(ref[:bus])])
# note: [i in keys(ref[:bus])] adds one `va` variable for each bus in the network

# Add active power generation variable pg for each generator
@variable(model, pg[i in keys(ref[:gen])])

# Add power flow variables p to represent the active power flow for each branch
@variable(model,  p[(l,i,j) in ref[:arcs_from]])

# Build JuMP expressions for the value of p[(l,i,j)] and p[(l,j,i)] on the branches
p_expr = Dict([((l,i,j), 1.0*p[(l,i,j)]) for (l,i,j) in ref[:arcs_from]])
p_expr = merge(p_expr, Dict([((l,j,i), -1.0*p[(l,i,j)]) for (l,i,j) in ref[:arcs_from]]))
# note: this is used to make the definition of nodal power balance simpler

# Add binary variables to represent that state of all components
@variable(model, z_bus[i in keys(ref[:bus])], Bin)
@variable(model, z_gen[i in keys(ref[:gen])], Bin)
@variable(model, z_branch[(l,i,j) in ref[:arcs_from]], Bin)

# Add continous load shed variable
@variable(model, 0.0 <= x_load[i in keys(ref[:load])] <= 1.0)

# variable_bus_active_indicator(pm)
# _PMR.variable_bus_voltage_on_off(pm)

# _PM.variable_gen_indicator(pm)
# _PM.variable_gen_power_on_off(pm)

# _PM.variable_branch_indicator(pm)
# _PM.variable_branch_power(pm)

# _PM.variable_load_power_factor(pm, relax=true)

_PMR.constraint_model_voltage_damage(pm)
for i in _PM.ids(pm, :ref_buses)
    _PM.constraint_theta_ref(pm, i)
end

for i in _PM.ids(pm, :gen)
    constraint_generation_active(pm, i)
    _PM.constraint_gen_power_on_off(pm, i)
end

for i in _PM.ids(pm, :bus)
    constraint_bus_active(pm, i)
    _PMR.constraint_power_balance_shed(pm, i)
end

for i in _PM.ids(pm, :branch)
    constraint_branch_active(pm, i)
    _PM.constraint_ohms_yt_from_on_off(pm, i)
    _PM.constraint_ohms_yt_to_on_off(pm, i)

    _PM.constraint_voltage_angle_difference_on_off(pm, i)

    _PM.constraint_thermal_limit_from_on_off(pm, i)
    _PM.constraint_thermal_limit_to_on_off(pm, i)
end

for i in _PM.ids(pm, :load)
    constraint_load_active(pm, i)
end


# Add Objective
# ------------------------------------
# Maximize power delivery while minimizing wildfire risk
z_demand = _PM.var(pm, nw_id_default, :z_demand)
# z_storage = _PM.var(pm, nw_id_default, :z_storage)
z_gen = _PM.var(pm, nw_id_default, :z_gen)
z_branch = _PM.var(pm, nw_id_default, :z_branch)
z_bus = _PM.var(pm, nw_id_default, :z_bus)

if haskey(_PM.ref(pm), :risk_weight)
    alpha = _PM.ref(pm, :risk_weight)
else
    Memento.warn(_PM._LOGGER, "network data should specify risk_weight, using 0.5 as a default")
    alpha = 0.5
end

for comp_type in [:gen, :load, :bus, :branch]
    for (id,comp) in  _PM.ref(pm, comp_type)
        if ~haskey(comp, "power_risk")
            Memento.warn(_PM._LOGGER, "$(comp_type) $(id) does not have a power_risk value, using 0.1 as a default")
            comp["power_risk"] = 0.1
        end
        if ~haskey(comp, "base_risk")
            Memento.warn(_PM._LOGGER, "$(comp_type) $(id) does not have a base_risk value, using 0.1 as a default")
            comp["base_risk"] = 0.1
        end
    end
end

JuMP.@objective(pm.model, Max,
    (1-alpha)*(
            sum(z_demand[i]*load["pd"] for (i,load) in _PM.ref(pm,:load))
    )
    - alpha*(
        sum(z_gen[i]*gen["power_risk"]+gen["base_risk"] for (i,gen) in _PM.ref(pm, :gen))
        + sum(z_bus[i]*bus["power_risk"]+bus["base_risk"] for (i,bus) in _PM.ref(pm, :bus))
        + sum(z_branch[i]*branch["power_risk"]+branch["base_risk"] for (i,branch) in _PM.ref(pm, :branch))
        + sum(z_demand[i]*load["power_risk"]+load["base_risk"] for (i,load) in _PM.ref(pm,:load))
        # + sum(z_storage[i]*storage["power_risk"]+storage["base_risk"] for (i,storage) in _PM.ref(pm, :storage))
    )
)

