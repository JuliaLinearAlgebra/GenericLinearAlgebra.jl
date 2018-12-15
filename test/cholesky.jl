using Test, LinearAlgebra, Random
using GenericLinearAlgebra: cholUnblocked!, cholBlocked!, cholRecursive!

@testset "Cholesky" begin

    Random.seed!(123)

    n = 50

    @testset "element type: $T" for T in (Float32, Float64, BigFloat)
        @testset "is complex: $cplx" for cplx in (true, false)
            if cplx
                A = complex.(T.(rand(n,n)), T.(rand(n,n)))
            else
                A = T.(rand(n,n))
            end
            AcA = A'A
            @test LowerTriangular(cholUnblocked!(copy(AcA), Val{:L}))   ≈ cholesky(Hermitian(AcA)).L
            @test LowerTriangular(cholBlocked!(copy(AcA), Val{:L}, 5))  ≈ cholesky(Hermitian(AcA)).L
            @test LowerTriangular(cholBlocked!(copy(AcA), Val{:L}, 10)) ≈ cholesky(Hermitian(AcA)).L
            @test LowerTriangular(cholRecursive!(copy(AcA), Val{:L}, 1)) ≈ cholesky(Hermitian(AcA)).L
            @test LowerTriangular(cholRecursive!(copy(AcA), Val{:L}, 4)) ≈ cholesky(Hermitian(AcA)).L
        end
    end
end