if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

using LinearAlgebra

@testset "The selfadjoint eigen problem" begin
    n = 10
    T = SymTridiagonal(randn(n), randn(n - 1))
    vals, vecs  = LinearAlgebra.EigenSelfAdjoint.eig(T)
    @test (vecs*Diagonal(vals))*vecs' ≈ full(T)
    vals, vecs2 = LinearAlgebra.EigenSelfAdjoint.eig2(T)
    @test vecs[[1,end],:] == vecs2
end
