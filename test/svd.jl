using Base.Test
using LinearAlgebra
using Quaternions

m = 6
n = 5

vals = reverse(collect(1:min(m,n)))
U = qr(Quaternion{Float64}[Quaternion(randn(4)...) for i = 1:m, j = 1:n])[1]
V = qr(Quaternion{Float64}[Quaternion(randn(4)...) for i = 1:n, j = 1:n])[1]

A = U*Diagonal(vals)*V'

@test vals ≈ LinearAlgebra.SVDModule.svdvals!(A)[1] # right now svdvals! also returns the iteration count. This should probably change
