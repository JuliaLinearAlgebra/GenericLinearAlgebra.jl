if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

@testset "The LinearAlgebra Test Suite" begin
    include("qr.jl")
    include("eigenselfadjoint.jl")
    include("eigengeneral.jl")
    include("tridiag.jl")
    include("svd.jl")
end
