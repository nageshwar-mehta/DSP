using Random, LinearAlgebra
include("../../trilat.jl")
# Reproduce same random seed
Random.seed!(123)
dim = 3; m = 10
# Generate data
s = randn(dim, m)
x_true = randn(dim)
# Compute squared distances
d2 = vec(sum((s .- x_true[:, ones(Int,m)]).^2, dims=1))
# Compute solution
x = trilat(s, d2)
# Print first solution as comma-separated values
println(join(x[:,1], ","))
