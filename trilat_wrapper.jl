# Activate project environment and ensure JSON
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()
Pkg.add("JSON")
using JSON, LinearAlgebra
include(joinpath(@__DIR__, "..", "..", "trilat.jl"))

# Read JSON input
input = JSON.parsefile(ARGS[1])
# sensors list-of-lists: input["s"] is s.T from Python => shape d×n
s_list = input["s"]
# Build matrix s of size d×n directly from columns
s = hcat([Vector{Float64}(col) for col in s_list]...)
# d2
d2 = Vector{Float64}(input["d2"])
# Compute and output first solution
x = trilat(s, d2)
# x is n×k; take first column
x1 = x[:,1]
# print JSON list
println(JSON.json(x1))
