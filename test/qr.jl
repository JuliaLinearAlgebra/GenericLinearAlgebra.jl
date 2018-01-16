using Base.Test
using LinearAlgebra
using LinearAlgebra.QRModule.qrBlocked!

@testset "The QR decomposition" begin
    @testset "Problem dimension ($m,$n) with block size $bz" for (m, n) in (( 10,  5), ( 10,  10), ( 5,  10),
                                                                            (100, 50), (100, 100), (50, 100)),
                                                                     bz in (1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33)

        A = randn(m, n)
        Aqr = qrBlocked!(copy(A), bz)
        AqrQ = Aqr[Tuple{:QBlocked}]
        if m >= n
            @test (AqrQ'A)[1:min(m,n),:] ≈ Aqr[Tuple{:R}]
        else # For type stbility getindex(,Tuple{:R}) throw when the output is trapezoid
            @test (AqrQ'A) ≈ triu(Aqr.factors)
        end
        @test AqrQ'*(AqrQ*A) ≈ A
    end

    @testset "Error paths" begin
        @test_throws DimensionMismatch LinearAlgebra.QRModule.reflectorApply!(zeros(5, 5), zeros(4), 1.0)
        @test_throws ArgumentError     qrBlocked!(randn(5, 10))[Tuple{:R}]
    end
end
