if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end
using LinearAlgebra

@testset "Test sign of eigenvalues" begin
    n = 20
    T = SymTridiagonal(randn(n), randn(n-1))
    @test numnegevals(T) == count(x->x<0, eigvals(T))
end

