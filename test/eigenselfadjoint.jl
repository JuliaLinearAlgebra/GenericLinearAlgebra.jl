using Base.Test
using LinearAlgebra

n = 100
T = SymTridiagonal(randn(n), randn(n - 1))
vals, vecs  = LinearAlgebra.EigenSelfAdjoint.eig(T)
@test_approx_eq vecs*diagm(vals)*vecs' full(T)
vals, vecs2 = LinearAlgebra.EigenSelfAdjoint.eig2(T)
@test vecs[[1,end],:] == vecs2
