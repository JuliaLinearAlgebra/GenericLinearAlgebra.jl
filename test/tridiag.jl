using GenericLinearAlgebra

@testset "Test sign of eigenvalues" begin
    n = 20
    T = SymTridiagonal(randn(n), randn(n - 1))
    @test numnegevals(T) == count(x -> x < 0, eigvals(T))
end
