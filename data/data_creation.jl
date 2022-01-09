# creating load/risk over time model

using Pkg; Pkg.activate(".")
using PowerModels
using DataFrames

case = parse_file(joinpath(@__DIR__,"case5_risk_sys1.m"))
periods = 24

fname = case["name"]

# write load sin_wave
open(joinpath(@__DIR__, "$(fname)_load.csv"),"w") do io
    print(io, "period")
    for (id,load) in case["load"]; print(io,",load_$id"); end
    println(io)

    for i in 1:periods
        print(io,"$i")
        for (id,load) in case["load"]
            print(io, ",$(load["pd"]*0.5+0.5*sin(2*pi*i/periods))")
        end
        println(io)
    end
end

# write risk cos wave
comp_list = String[]
for comp_type in ["bus","branch","gen","load"]
    append!(comp_list, ["$(comp_type)_$(id)" for id in keys(case[comp_type])])
end


open(joinpath(@__DIR__, "$(fname)_risk.csv"),"w") do io
    print(io, "period")
    for comp in comp_list ; print(io,",$comp"); end
    println(io)

    for i in 1:periods
        print(io,"$i")
        for comp in comp_list
            print(io, ",$(rand()+0.5*cos(2*pi*i/periods))")
        end
        println(io)
    end
end




