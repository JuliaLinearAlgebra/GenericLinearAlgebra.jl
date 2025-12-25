using Test, LinearAlgebra, Random, Quaternions

Random.seed!(123)

@testset "ldlt. n=$n, uplo=$uplo" for n in [5, 50, 500], uplo in [:U, :L]
    A = [Quaternion(randn(4)...) for i in 1:n, j in 1:(n - 2)]
    Brr = Hermitian(A * A', uplo)
    o = ones(n)

    @testset "basics" begin
        Frr = ldlt(Brr)
        @test Frr.L*Frr.D*Frr.L' ≈ Brr
    end

    @testset "solves" begin
        Bfr = Brr + I
        Ffr = ldlt(Bfr)
        @test Bfr*(Ffr\o) ≈ o
        @test Ffr*(Ffr\o) ≈ o
    end
end
