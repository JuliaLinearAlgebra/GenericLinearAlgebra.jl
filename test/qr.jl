if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

using Base.LAPACK
using LinearAlgebra
using LinearAlgebra.QRModule.qrUnblocked!


@testset "The QR decomposition. Problem dimension ($m,$n)" for (m, n) = ((10, 5), (10, 10), (5, 10))

    A = randn(m, n)
    Aqr = qrUnblocked!(copy(A))
    AqrQ = Aqr[Tuple{:QBlocked}]
    if m >= n
        @test (AqrQ'A)[1:min(m,n),:] ≈ Aqr[Tuple{:R}]
    else # For type stbility getindex(,Tuple{:R}) throw when the output is trapezoid
        @test (AqrQ'A) ≈ triu(Aqr.factors)
    end
    @test AqrQ'*(AqrQ*A) ≈ A
end
