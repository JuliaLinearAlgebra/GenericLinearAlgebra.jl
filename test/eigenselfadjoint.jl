using LinearAlgebra

@testset "The selfadjoint eigen problem" begin
    n = 10
    @testset "SymTridiagonal" begin
        T = SymTridiagonal(randn(n), randn(n - 1))
        vals, vecs  = LinearAlgebra.EigenSelfAdjoint.eig(T)
        @test (vecs*Diagonal(vals))*vecs' ≈ full(T)
        vals, vecs2 = LinearAlgebra.EigenSelfAdjoint.eig2(T)
        @test vecs[[1,end],:] == vecs2
    end

    @testset "(full) Symmetric" begin
        A = Symmetric(randn(n, n), :L)
        vals, vecs = LinearAlgebra.EigenSelfAdjoint.eig(A)
        @test vecs'*A*vecs ≈ diagm(vals)
    end
end
