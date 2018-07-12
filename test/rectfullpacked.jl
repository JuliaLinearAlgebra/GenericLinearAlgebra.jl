using Test
import GenericLinearAlgebra
import GenericLinearAlgebra: Ac_mul_A_RFP, TriangularRFP

@testset "Rectuangular Full Pack Format" begin

    @testset "Core generic functionality" for n in (6, 7),
                                           uplo in (:U, :L)

        A = rand(10, n)

        @testset "Hermitian" begin
            AcA_RFP = Ac_mul_A_RFP(A, uplo)

            @test size(AcA_RFP, 1) == n
            @test size(AcA_RFP, 2) == n
            @test size(AcA_RFP, 3) == 1
            @test_throws BoundsError AcA_RFP[0, 1]
            @test_throws BoundsError AcA_RFP[1, 0]
            @test_throws BoundsError AcA_RFP[n + 1, 1]
            @test_throws BoundsError AcA_RFP[1, n + 1]

            @test AcA_RFP[2, 1]             == AcA_RFP[1, 2]
            @test AcA_RFP[end - 2, end - 1] == AcA_RFP[end - 1, end - 2]
        end

        @testset "Triangular" begin
            Atr_RFP = TriangularRFP(triu(A'A), uplo)
            @test size(Atr_RFP, 1) == n
            @test size(Atr_RFP, 2) == n
            @test size(Atr_RFP, 3) == 1

            # Indexing not implemented yet for Atr_RFP
            # @test_throws BoundsError Atr_RFP[0, 1]
            # @test_throws BoundsError Atr_RFP[1, 0]
            # @test_throws BoundsError Atr_RFP[n + 1, 1]
            # @test_throws BoundsError Atr_RFP[1, n + 1]

            @test_broken Atr_RFP[2, 1]             == Atr_RFP[1, 2]
            @test_broken Atr_RFP[end - 2, end - 1] == Atr_RFP[end - 1, end - 2]
        end
    end

    @testset "Hermitian with element type: $elty. Problem size: $n" for elty in (Float32, Float64, Complex{Float32}, Complex{Float64}),
                                                                           n in (6, 7),
                                                                        uplo in (:L, :U)

        A       = rand(elty, 10, n)
        AcA     = A'A
        AcA_RFP = Ac_mul_A_RFP(A, uplo)
        o       = ones(elty, n)

        @test AcA   ≈ A'A
        @test AcA\o ≈ AcA_RFP\o
        @test inv(AcA) ≈ inv(AcA_RFP)
        @test inv(cholfact(AcA)) ≈ inv(factorize(AcA_RFP))
    end

    @testset "Hermitian with element type: $elty. Problem size: $n" for elty in (Float32, Float64, Complex{Float32}, Complex{Float64}),
                                                                           n in (6, 7),
                                                                        uplo in (:L, :U)

        A     = lufact(rand(elty, n, n))[:U]
        A     = uplo == :U ? A : A'
        A_RFP = TriangularRFP(A, uplo)
        o     = ones(elty, n)

        @test_broken A ≈ A_RFP
        @test        A ≈ full(A_RFP)
        @test      A\o ≈ A_RFP\o
        @test   inv(A) ≈ full(inv(A_RFP))
    end
end
