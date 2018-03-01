using Base.Test, GenericLinearAlgebra
using LinearAlgebra.CholeskyModule: cholUnblocked!, cholBlocked!, cholRecursive!

n = 50

@testset "Cholesky" begin
    @testset "element type: $T" for T in (Float32, Float64, BigFloat)
        @testset "is complex: $cplx" for cplx in (true, false)
            if cplx
                A = complex.(T.(rand(n,n)), T.(rand(n,n)))
            else
                A = T.(rand(n,n))
            end
            AcA = A'A
            @test LowerTriangular(cholUnblocked!(copy(AcA), Val{:L}))   ≈ chol(Hermitian(AcA))'
            @test LowerTriangular(cholBlocked!(copy(AcA), Val{:L}, 5))  ≈ chol(Hermitian(AcA))'
            @test LowerTriangular(cholBlocked!(copy(AcA), Val{:L}, 10)) ≈ chol(Hermitian(AcA))'
            @test LowerTriangular(cholRecursive!(copy(AcA), Val{:L}, 1)) ≈ chol(Hermitian(AcA))'
            @test LowerTriangular(cholRecursive!(copy(AcA), Val{:L}, 4)) ≈ chol(Hermitian(AcA))'
        end
    end
end
