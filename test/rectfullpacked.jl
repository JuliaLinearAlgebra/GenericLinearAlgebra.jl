using Base.Test
using LinearAlgebra

@testset "Rectuangular Full Pack Format" begin

    @testset "Element type: $elty. Problem size: $n" for elty in (Float32, Float64, Complex{Float32}, Complex{Float64}),
        n in (6, 7)

        A       = rand(elty, 10, n)
        AcA     = A'A
        AcA_RFP = LinearAlgebra.Ac_mul_A_RFP(A)
        o       = ones(elty, n)

        @test AcA   ≈ A'A
        @test AcA\o ≈ AcA_RFP\o
        @test inv(AcA) ≈ inv(AcA_RFP)
    end
end
