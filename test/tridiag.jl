using Base.Test
using LinearAlgebra

n = 20
T = SymTridiagonal(randn(n), randn(n-1))
@test numnegevals(T) == count(x->x<0, eigvals(T))

