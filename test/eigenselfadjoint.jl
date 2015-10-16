using Base.Test
using LinearAlgebra

n = 10
T = SymTridiagonal(randn(n), randn(n - 1))
vals, vecs  = LinearAlgebra.EigenSelfAdjoint.eig(T)
@test (vecs*Diagonal(vals))*vecs' ≈ full(T)
vals, vecs2 = LinearAlgebra.EigenSelfAdjoint.eig2(T)
@test vecs[[1,end],:] == vecs2
