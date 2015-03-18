using Base.Test
using Base.LAPACK
using LinearAlgebra
using LinearAlgebra.QRModule.qrUnblocked!

A = randn(10,5)
Aqr = qrUnblocked!(copy(A))
@test_approx_eq A Aqr[Val{:Q}]*Aqr[Val{:R}]
