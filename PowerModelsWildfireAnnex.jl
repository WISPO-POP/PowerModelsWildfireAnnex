
# import JuMP
# import JuMP: @variable, @constraint, @NLexpression, @NLconstraint, @objective, @NLobjective, @expression, optimize!, Model
# import InfrastructureModels; const _IM = InfrastructureModels
# import PowerModels; const _PM = PowerModels
# import PowerModels: ids, ref, var, con, sol, nw_id_default
# import Memento
# const LOGGER = Memento.getlogger(PowerModels)


#### DC Optimal Power Shutoff ####

# This file provides a basic implentation of the DC Optimal Power Shutoff
# problem for experimentation on algorithm and problem variations.

# This file can be run by calling `include("dc-ops.jl")` from the Julia REPL

# Developed by Noah Rhodes(@noahrhodes) and Line Roald (@lroald)


###############################################################################
# 0. Initialization
###############################################################################
using Pkg; Pkg.activate(".")
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
file_name = "$(powermodels_path)/test/networks/RTS_GMLC_risk.m"
# file_name = "$(powermodels_path)/test/networks/case14_risk.m"
# file_name = "$(powermodels_path)/test/networks/case5_risk_sys1.m"
# file_name = "$(powermodels_path)/test/networks/case5_risk_sys2.m "

# load the data file
data = PowerModels.parse_file(file_name)



# Modify risk weighting
# data["risk_weight"] = 0.2

# Add zeros to turn linear objective functions into quadratic ones
# so that additional parameter checks are not required
PowerModels.standardize_cost_terms!(data, order=2)

# Adds reasonable rate_a values to branches without them
PowerModels.calc_thermal_limits!(data)

# use build_ref to filter out inactive components
ref = PowerModels.build_ref(data)
ref_add_on_off_va_bounds!(ref, data)
ref = ref[:it][:pm][:nw][0]

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
@variable(model, 0.0 <= x_load[l in keys(ref[:load])] <= 1.0)
@variable(model, 0.0 <= x_shunt[s in keys(ref[:shunt])] <= 1.0)


# Add Objective Function
# ----------------------
# Maximize power delivery while minimizing wildfire risk

if haskey(ref, :risk_weight)
    alpha = ref[:risk_weight]
else
    @warn "network data should specify risk_weight, using 0.5 as a default"
    alpha = 0.5
end

for comp_type in [:gen, :load, :bus, :branch]
    for (id,comp) in  ref[comp_type]
        if ~haskey(comp, "power_risk")
            @warn "$(comp_type) $(id) does not have a power_risk value, using 0.1 as a default"
            comp["power_risk"] = 0.1
        end
    end
end

JuMP.@objective(model, Max,
    (1-alpha)*(
            sum(x_load[i]*load["pd"] for (i,load) in ref[:load])
    )
    - alpha*(
        sum(z_gen[i]*gen["power_risk"] for (i,gen) in ref[:gen])
        + sum(z_bus[i]*bus["power_risk"] for (i,bus) in ref[:bus])
        + sum(z_branch[(l,i,j)]*ref[:branch][l]["power_risk"] for (l,i,j) in ref[:arcs_from])
        + sum(x_load[i]*load["power_risk"] for (i,load) in ref[:load])
    )
)


# Add Constraints
# ---------------

# Fix the voltage angle to zero at the reference bus
for (i,bus) in ref[:ref_buses]
    @constraint(model, va[i] == 0)
end

# Nodal power balance constraints
for (i,bus) in ref[:bus]
    # Build a list of the loads and shunt elements connected to the bus i
    bus_loads = [(l,ref[:load][l]) for l in ref[:bus_loads][i]]
    bus_shunts = [(s,ref[:shunt][s]) for s in ref[:bus_shunts][i]]

    # Active power balance at node i
    @constraint(model,
        sum(p_expr[a] for a in ref[:bus_arcs][i]) ==                    # sum of active power flow on lines from bus i +
        sum(pg[g] for g in ref[:bus_gens][i]) -                         # sum of active power generation at bus i -
        sum(x_load[l]*load["pd"] for (l,load) in bus_loads) -           # sum of active load * load shed  at bus i -
        sum(x_shunt[s]*shunt["gs"] for (s,shunt) in bus_shunts)*1.0^2   # sum of active shunt element injections * shed at bus i
    )
end

# Branch power flow physics and limit constraints
for (i,branch) in ref[:branch]
    # Build the from variable id of the i-th branch, which is a tuple given by (branch id, from bus, to bus)
    f_idx = (i, branch["f_bus"], branch["t_bus"])
    p_fr = p[f_idx]                     # p_fr is a reference to the optimization variable p[f_idx]
    z_br = z_branch[f_idx]
    va_fr = va[branch["f_bus"]]         # va_fr is a reference to the optimization variable va on the from side of the branch
    va_to = va[branch["t_bus"]]         # va_fr is a reference to the optimization variable va on the to side of the branch
    # Compute the branch parameters and transformer ratios from the data
    g, b = PowerModels.calc_branch_y(branch)

    # Voltage angle difference limit
    JuMP.@constraint(model, va_fr - va_to <= branch["angmax"]*z_br + ref[:off_angmax]*(1-z_br))
    JuMP.@constraint(model, va_fr - va_to >= branch["angmin"]*z_br + ref[:off_angmin]*(1-z_br))

    # DC Power Flow Constraint
    if b <= 0
        JuMP.@constraint(model, p_fr <= -b*(va_fr - va_to + ref[:off_angmax]*(1-z_br)) )
        JuMP.@constraint(model, p_fr >= -b*(va_fr - va_to + ref[:off_angmin]*(1-z_br)) )
    else # account for bound reversal when b is positive
        JuMP.@constraint(model, p_fr >= -b*(va_fr - va_to + ref[:off_angmax]*(1-z_br)) )
        JuMP.@constraint(model, p_fr <= -b*(va_fr - va_to + ref[:off_angmin]*(1-z_br)) )
    end

    # Thermal limit
    JuMP.@constraint(model, p_fr <=  branch["rate_a"]*z_br)
    JuMP.@constraint(model, p_fr >= -branch["rate_a"]*z_br)

    # Connectivity constraint
    JuMP.@constraint(model, z_br <= z_bus[branch["f_bus"]])
    JuMP.@constraint(model, z_br <= z_bus[branch["t_bus"]])
end

# Generator constraints
for (i,gen) in ref[:gen]
    # Power limit
    JuMP.@constraint(model, pg[i] <= gen["pmax"]*z_gen[i])
    JuMP.@constraint(model, pg[i] >= gen["pmin"]*z_gen[i])

    # Connectivity constraint
    JuMP.@constraint(model, z_gen[i] <= z_bus[gen["gen_bus"]])
end

# Load constraints
for (i, load) in ref[:load]
    # Connectivity constraints
    JuMP.@constraint(model, x_load[i] <= z_bus[load["load_bus"]])
end

###############################################################################
# 3. Solve the Optimal Power Flow Model and Review the Results
###############################################################################

# Solve the optimization problem
optimize!(model)

# Check that the solver terminated without an error
println("The solver termination status is $(termination_status(model))")

# Check the value of the objective function
println("The objective value is $(objective_value(model))")
total_load = sum(value(x_load[i])*load["pd"] for (i,load) in ref[:load])
println("The load served is $total_load ")
total_risk = sum(value(z_branch[(l,i,j)])*ref[:branch][l]["power_risk"] for (l,i,j) in ref[:arcs_from]) +
             sum(value(z_gen[g])*gen["power_risk"] for (g,gen) in ref[:gen]) +
             sum(value(z_bus[i])*bus["power_risk"] for (i,bus) in ref[:bus]) +
             sum(value(x_load[i])*load["power_risk"] for (i,load) in ref[:load])
println("The system risk is $total_risk")

# Get inactive devices
for l in sort(collect(keys(ref[:branch])))
    f_idx = (l, ref[:branch][l]["f_bus"], ref[:branch][l]["t_bus"])
    value(z_branch[f_idx])==0 ? println("Branch $l is disabled.") : false
end
for g in sort(collect(keys(ref[:gen])))
    value(z_gen[g])==0 ? println("Generator $g is disabled.") : false
end
for i in sort(collect(keys(ref[:bus])))
    value(z_bus[i])==0 ? println("Bus $i is disabled.") : false
end
for i in sort(collect(keys(ref[:load])))
    value(x_load[i])!=1 ? println("Load $i is serving $(value(x_load[i])*100)% of demand.") : false
end