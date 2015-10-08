using Base.Test
using Base.LAPACK
using LinearAlgebra
using LinearAlgebra.QRModule.qrUnblocked!

m, n = 10, 5

A = randn(m, n)
Aqr = qrUnblocked!(copy(A))
AqrQ = Aqr[Tuple{:QBlocked}]
@test AqrQ'A ≈ [Aqr[Tuple{:R}]; zeros(m - n, n)]
@test AqrQ'*(AqrQ*A) ≈ A
