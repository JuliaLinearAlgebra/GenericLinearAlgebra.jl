using Test, Random, GenericLinearAlgebra
using LinearAlgebra: LinearAlgebra

@testset "Cholesky" begin

    Random.seed!(123)

    n = 50

    @testset "element type: $T" for T in (Float32, Float64, BigFloat)
        @testset "is complex: $cplx" for cplx in (true, false)
            if cplx
                A = complex.(T.(rand(n, n)), T.(rand(n, n)))
            else
                A = T.(rand(n, n))
            end
            AcA = A'A
            @test LinearAlgebra.LowerTriangular(GenericLinearAlgebra.cholUnblocked!(copy(AcA), Val{:L})) ≈
                  LinearAlgebra.cholesky(LinearAlgebra.Hermitian(AcA)).L
            @test LinearAlgebra.LowerTriangular(GenericLinearAlgebra.cholBlocked!(copy(AcA), Val{:L}, 5)) ≈
                  LinearAlgebra.cholesky(LinearAlgebra.Hermitian(AcA)).L
            @test LinearAlgebra.LowerTriangular(GenericLinearAlgebra.cholBlocked!(copy(AcA), Val{:L}, 10)) ≈
                  LinearAlgebra.cholesky(LinearAlgebra.Hermitian(AcA)).L
            @test LinearAlgebra.LowerTriangular(GenericLinearAlgebra.cholRecursive!(copy(AcA), Val{:L}, 1)) ≈
                  LinearAlgebra.cholesky(LinearAlgebra.Hermitian(AcA)).L
            @test LinearAlgebra.LowerTriangular(GenericLinearAlgebra.cholRecursive!(copy(AcA), Val{:L}, 4)) ≈
                  LinearAlgebra.cholesky(LinearAlgebra.Hermitian(AcA)).L
        end
    end
end
