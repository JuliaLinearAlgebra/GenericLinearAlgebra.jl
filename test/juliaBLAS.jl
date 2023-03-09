using Test, GenericLinearAlgebra, LinearAlgebra

@testset "rankUpdate!" begin
    A, B, x = (Hermitian(randn(5, 5)), randn(5, 2), randn(5))
    Ac, Bc, xc = (
        Hermitian(complex.(randn(5, 5), randn(5, 5))),
        complex.(randn(5, 2), randn(5, 2)),
        complex.(randn(5), randn(5)),
    )
    @test rankUpdate!(copy(A), x) ≈ A .+ x .* x'
    @test rankUpdate!(copy(Ac), xc) ≈ Ac .+ xc .* xc'

    @test rankUpdate!(copy(A), B, 0.5, 0.5) ≈ 0.5 * A + 0.5 * B * B'
    @test rankUpdate!(copy(Ac), Bc, 0.5, 0.5) ≈ 0.5 * Ac + 0.5 * Bc * Bc'

    @test invoke(rankUpdate!, Tuple{Hermitian,StridedVecOrMat,Real}, copy(Ac), Bc, 1.0) ≈
          rankUpdate!(copy(Ac), Bc, 1.0, 1.0)
end

@testset "triangular multiplication: $(typeof(T))" for T ∈ (
    UpperTriangular(complex.(randn(5, 5), randn(5, 5))),
    UnitUpperTriangular(complex.(randn(5, 5), randn(5, 5))),
    LowerTriangular(complex.(randn(5, 5), randn(5, 5))),
    UnitLowerTriangular(complex.(randn(5, 5), randn(5, 5))),
)
    B = complex.(randn(5, 5), randn(5, 5))
    @test lmul!(T, copy(B), complex(0.5, 0.5)) ≈ T * B * complex(0.5, 0.5)
    @test lmul!(T', copy(B), complex(0.5, 0.5)) ≈ T' * B * complex(0.5, 0.5)
end
