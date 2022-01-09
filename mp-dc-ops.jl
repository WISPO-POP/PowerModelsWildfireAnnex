
#### Multiperiod DC Optimal Power Shutoff ####

# This file provides a basic implentation of the multiperiod DC Optimal Power Shutoff
# problem for experimentation on algorithm and problem variations.

# This file can be run by calling `include("mp-dc-ops.jl")` from the Julia REPL

# Developed by Noah Rhodes(@noahrhodes) and Line Roald (@lroald)

###############################################################################
# 0. Initialization
###############################################################################
using Pkg; Pkg.activate(".")
Pkg.instantiate()
Pkg.precompile()

# Load Julia Packages
#--------------------
using PowerModels
using PowerModelsWildfire
using Gurobi
using JuMP
using CSV
using DataFrames

# Load System Data
# ----------------

# load the data file
data = PowerModels.parse_file(joinpath(@__DIR__, "data", "case5_risk_sys1.m"))
load_data = CSV.read(joinpath(@__DIR__, "data", "case5_load.csv"),DataFrame)
risk_data = CSV.read(joinpath(@__DIR__, "data", "case5_risk.csv"),DataFrame)

# Modify risk weighting
# data["risk_weight"] = 0.2

# Add zeros to turn linear objective functions into quadratic ones
# so that additional parameter checks are not required
PowerModels.standardize_cost_terms!(data, order=2)

# Adds reasonable rate_a values to branches without them
PowerModels.calc_thermal_limits!(data)


# Create multiperiod data model
mp_data = PowerModels.replicate(data, size(load_data,1))

for row in eachrow(load_data)
    for col in names(row[2:end])
        col
        nwid = string(row[1])
        comp_type, comp_id = split(col,"_")
        load = row[col]

        mp_data["nw"][nwid][comp_type][comp_id]["pd"] = load
    end
end

for row in eachrow(risk_data)
    for col in names(row[2:end])
        col
        nwid = string(row[1])
        comp_type, comp_id = split(col,"_")
        risk = row[col]

        mp_data["nw"][nwid][comp_type][comp_id]["power_risk"] = risk
    end
end


# use build_ref to filter out inactive components
ref = PowerModels.build_ref(mp_data)
ref_add_on_off_va_bounds!(ref, mp_data)
ref = ref[:it][:pm]

# Note: ref contains all the relevant system parameters needed to build the OPS model
# When we introduce constraints and variable bounds below, we use the parameters in ref.


###############################################################################
# 1. Building the Optimal Power Shutoff Model
###############################################################################

# Initialize a JuMP Optimization Model
#-------------------------------------
# model = Model(GLPK.Optimizer)
model = Model(Gurobi.Optimizer)

# Add Optimization and State Variables
# ------------------------------------

# Add voltage angles va for each bus
@variable(model, va[t in keys(ref[:nw]),i in keys(ref[:nw][1][:bus])])
# note: [i in keys(ref[:nw][1][:bus])] adds one `va` variable for each bus in the network

# Add active power generation variable pg for each generator
@variable(model, pg[t in keys(ref[:nw]),i in keys(ref[:nw][1][:gen])])

# Add power flow variables p to represent the active power flow for each branch
@variable(model,  p[t in keys(ref[:nw]), (l,i,j) in ref[:nw][1][:arcs_from]])

# Build JuMP expressions for the value of p[(l,i,j)] and p[(l,j,i)] on the branches
p_expr = Dict([((t,(l,i,j)), 1.0*p[t,(l,i,j)]) for (l,i,j) in ref[:nw][1][:arcs_from], t in keys(ref[:nw])])
p_expr = merge(p_expr, Dict([((t,(l,j,i)), -1.0*p[t,(l,i,j)]) for (l,i,j) in ref[:nw][1][:arcs_from], t in keys(ref[:nw])]))
# note: this is used to make the definition of nodal power balance simpler

# Add binary variables to represent that state of all components
@variable(model, z_bus[t in keys(ref[:nw]), i in keys(ref[:nw][1][:bus])], Bin)
@variable(model, z_gen[t in keys(ref[:nw]), i in keys(ref[:nw][1][:gen])], Bin)
@variable(model, z_branch[t in keys(ref[:nw]), (l,i,j) in ref[:nw][1][:arcs_from]], Bin)

# Add continous load shed variable
@variable(model, 0.0 <= x_load[t in keys(ref[:nw]), l in keys(ref[:nw][1][:load])] <= 1.0)
@variable(model, 0.0 <= x_shunt[t in keys(ref[:nw]), s in keys(ref[:nw][1][:shunt])] <= 1.0)


# Add Objective Function
# ----------------------
# Maximize power delivery while minimizing wildfire risk

if haskey(ref[:nw][1], :risk_weight)
    alpha = ref[:nw][1][:risk_weight]
else
    @warn "network data should specify risk_weight, using 0.5 as a default"
    alpha = 0.5
end

for comp_type in [:gen, :load, :bus, :branch]
    for (nwid,nw) in ref[:nw]
        for (id,comp) in  nw[comp_type]
            if ~haskey(comp, "power_risk")
                @warn "$(comp_type) $(id) does not have a power_risk value, using 0.1 as a default"
                comp["power_risk"] = 0.1
            end
        end
    end
end

JuMP.@objective(model, Max,
    sum(
        (1-alpha)*(
                sum(x_load[t,i]*load["pd"] for (i,load) in nw[:load])
        )
        - alpha*(
            sum(z_gen[t,i]*gen["power_risk"] for (i,gen) in nw[:gen])
            + sum(z_bus[t,i]*bus["power_risk"] for (i,bus) in nw[:bus])
            + sum(z_branch[t,(l,i,j)]*nw[:branch][l]["power_risk"] for (l,i,j) in nw[:arcs_from])
            + sum(x_load[t,i]*load["power_risk"] for (i,load) in nw[:load])
        )
        for (t,nw) in ref[:nw]
    )
)


# Add Constraints
# ---------------

#  constraints for each period of OPS
for (t,nw) in ref[:nw]

    # Fix the voltage angle to zero at the reference bus
    for (i,bus) in nw[:ref_buses]
        @constraint(model, va[t,i] == 0)
    end

    # Nodal power balance constraints
    for (i,bus) in nw[:bus]
        # Build a list of the loads and shunt elements connected to the bus i
        bus_loads = [(l,nw[:load][l]) for l in nw[:bus_loads][i]]
        bus_shunts = [(s,nw[:shunt][s]) for s in nw[:bus_shunts][i]]

        # Active power balance at node i
        @constraint(model,
            sum(p_expr[(t,a)] for a in nw[:bus_arcs][i]) ==                    # sum of active power flow on lines from bus i +
            sum(pg[t,g] for g in nw[:bus_gens][i]) -                         # sum of active power generation at bus i -
            sum(x_load[t,l]*load["pd"] for (l,load) in bus_loads) -           # sum of active load * load shed  at bus i -
            sum(x_shunt[t,s]*shunt["gs"] for (s,shunt) in bus_shunts)*1.0^2   # sum of active shunt element injections * shed at bus i
        )
    end

    # Branch power flow physics and limit constraints
    for (i,branch) in nw[:branch]
        # Build the from variable id of the i-th branch, which is a tuple given by (branch id, from bus, to bus)
        f_idx = (i, branch["f_bus"], branch["t_bus"])
        p_fr = p[t,f_idx]                     # p_fr is a reference to the optimization variable p[f_idx]
        z_br = z_branch[t,f_idx]
        va_fr = va[t,branch["f_bus"]]         # va_fr is a reference to the optimization variable va on the from side of the branch
        va_to = va[t,branch["t_bus"]]         # va_fr is a reference to the optimization variable va on the to side of the branch
        # Compute the branch parameters and transformer ratios from the data
        g, b = PowerModels.calc_branch_y(branch)

        # Voltage angle difference limit
        JuMP.@constraint(model, va_fr - va_to <= branch["angmax"]*z_br + nw[:off_angmax]*(1-z_br))
        JuMP.@constraint(model, va_fr - va_to >= branch["angmin"]*z_br + nw[:off_angmin]*(1-z_br))

        # DC Power Flow Constraint
        if b <= 0
            JuMP.@constraint(model, p_fr <= -b*(va_fr - va_to + nw[:off_angmax]*(1-z_br)) )
            JuMP.@constraint(model, p_fr >= -b*(va_fr - va_to + nw[:off_angmin]*(1-z_br)) )
        else # account for bound reversal when b is positive
            JuMP.@constraint(model, p_fr >= -b*(va_fr - va_to + nw[:off_angmax]*(1-z_br)) )
            JuMP.@constraint(model, p_fr <= -b*(va_fr - va_to + nw[:off_angmin]*(1-z_br)) )
        end

        # Thermal limit
        JuMP.@constraint(model, p_fr <=  branch["rate_a"]*z_br)
        JuMP.@constraint(model, p_fr >= -branch["rate_a"]*z_br)

        # Connectivity constraint
        JuMP.@constraint(model, z_br <= z_bus[t,branch["f_bus"]])
        JuMP.@constraint(model, z_br <= z_bus[t,branch["t_bus"]])
    end

    # Generator constraints
    for (i,gen) in nw[:gen]
        # Power limit
        JuMP.@constraint(model, pg[t,i] <= gen["pmax"]*z_gen[t,i])
        JuMP.@constraint(model, pg[t,i] >= gen["pmin"]*z_gen[t,i])

        # Connectivity constraint
        JuMP.@constraint(model, z_gen[t,i] <= z_bus[t,gen["gen_bus"]])
    end

    # Load constraints
    for (i, load) in nw[:load]
        # Connectivity constraints
        JuMP.@constraint(model, x_load[t,i] <= z_bus[t,load["load_bus"]])
    end
end

# Inter-period constraints
for t in keys(ref[:nw])
    if t != 1
        # if component is disabled, it must stay disabled for future periods
        for (i,branch) in ref[:nw][t][:branch]
            f_idx = (i, branch["f_bus"], branch["t_bus"])
            JuMP.@constraint(model, z_branch[t-1,f_idx] >= z_branch[t,f_idx])
        end

        for (i,bus) in ref[:nw][t][:bus]
            JuMP.@constraint(model, z_bus[t-1,i] >= z_bus[t,i])
        end
        for (i,gen) in ref[:nw][t][:gen]
            JuMP.@constraint(model, z_gen[t-1,i] >= z_gen[t,i])
        end
    end
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
total_load = sum(sum(value(x_load[t,i])*load["pd"] for (i,load) in nw[:load]) for (t,nw) in ref[:nw])
println("The load served is $total_load ")
total_risk = sum(
    sum(value(z_branch[t,(l,i,j)])*nw[:branch][l]["power_risk"] for (l,i,j) in nw[:arcs_from]) +
    sum(value(z_gen[t,g])*gen["power_risk"] for (g,gen) in nw[:gen]) +
    sum(value(z_bus[t,i])*bus["power_risk"] for (i,bus) in nw[:bus]) +
    sum(value(x_load[t,i])*load["power_risk"] for (i,load) in nw[:load])
    for (t,nw) in ref[:nw]
)
println("The system risk is $total_risk")

# Get inactive devices
for t in sort(collect(keys(ref[:nw])))
    nw = ref[:nw][t]
    for l in sort(collect(keys(nw[:branch])))
        f_idx = (l, nw[:branch][l]["f_bus"], nw[:branch][l]["t_bus"])
        value(z_branch[t,f_idx])==0 ? println("Branch $l is disabled in period $t.") : false
    end
    for g in sort(collect(keys(nw[:gen])))
        value(z_gen[t,g])==0 ? println("Generator $g is disabled in period $t.") : false
    end
    for i in sort(collect(keys(nw[:bus])))
        value(z_bus[t,i])==0 ? println("Bus $i is disabled in period $t.") : false
    end
    for i in sort(collect(keys(nw[:load])))
        value(x_load[t,i])!=1 ? println("Load $i is serving $(value(x_load[t,i])*100)% of demand in period $t.") : false
    end
end