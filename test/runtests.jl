using Test

# @testset "The LinearAlgebra Test Suite" begin
include("juliaBLAS.jl")
include("cholesky.jl")
include("qr.jl")
include("eigenselfadjoint.jl")
include("eigengeneral.jl")
include("tridiag.jl")
include("svd.jl")
include("rectfullpacked.jl")
include("lapack.jl")
# end
