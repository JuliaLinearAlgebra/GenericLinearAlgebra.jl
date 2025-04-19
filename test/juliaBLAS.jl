using Test, GenericLinearAlgebra
using LinearAlgebra: LinearAlgebra

@testset "rankUpdate!" begin
    A, B, x = (LinearAlgebra.Hermitian(randn(5, 5)), randn(5, 2), randn(5))
    Ac, Bc, xc = (
        LinearAlgebra.Hermitian(complex.(randn(5, 5), randn(5, 5))),
        complex.(randn(5, 2), randn(5, 2)),
        complex.(randn(5), randn(5)),
    )
    @test GenericLinearAlgebra.rankUpdate!(copy(A), x) ≈ A .+ x .* x'
    @test GenericLinearAlgebra.rankUpdate!(copy(Ac), xc) ≈ Ac .+ xc .* xc'

    @test GenericLinearAlgebra.rankUpdate!(copy(A), B, 0.5, 0.5) ≈ 0.5 * A + 0.5 * B * B'
    @test GenericLinearAlgebra.rankUpdate!(copy(Ac), Bc, 0.5, 0.5) ≈ 0.5 * Ac + 0.5 * Bc * Bc'

    @test GenericLinearAlgebra.rankUpdate!(copy(Ac), Bc, 1.0) ≈
          GenericLinearAlgebra.rankUpdate!(copy(Ac), Bc, 1.0, 1.0)
end

@testset "triangular multiplication: $(typeof(T))" for T ∈ (
    LinearAlgebra.UpperTriangular(complex.(randn(5, 5), randn(5, 5))),
    LinearAlgebra.UnitUpperTriangular(complex.(randn(5, 5), randn(5, 5))),
    LinearAlgebra.LowerTriangular(complex.(randn(5, 5), randn(5, 5))),
    LinearAlgebra.UnitLowerTriangular(complex.(randn(5, 5), randn(5, 5))),
)
    B = complex.(randn(5, 5), randn(5, 5))
    @test GenericLinearAlgebra.lmul!(T, copy(B), complex(0.5, 0.5)) ≈ T * B * complex(0.5, 0.5)
    @test GenericLinearAlgebra.lmul!(T', copy(B), complex(0.5, 0.5)) ≈ T' * B * complex(0.5, 0.5)
end
