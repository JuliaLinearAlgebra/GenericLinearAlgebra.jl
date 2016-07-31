using Base.Test
using LinearAlgebra
using Quaternions

let m = 6, n = 5

    vals = reverse(collect(1:min(m,n)))
    U = qr(Quaternion{Float64}[Quaternion(randn(4)...) for i = 1:m, j = 1:n])[1]
    V = qr(Quaternion{Float64}[Quaternion(randn(4)...) for i = 1:n, j = 1:n])[1]

    A = U*Diagonal(vals)*V'

    @test vals ≈ svdvals(A)
end

# This matrix used to hang (for n = 70). Thanks to Ivan Slapnicar for reporting.
let n = 70
    J = Bidiagonal(0.5 * ones(n), ones(n-1), true)
    @test LinearAlgebra.SVDModule.svdvals!(copy(J)) ≈ svdvals(J)
    @test LinearAlgebra.SVDModule.svdvals!(copy(J))[end] / svdvals(J)[end] - 1 < n*eps()
end

# Test that we extend the base method correctly
## Tall case
for (m, n) in ((10,9), # tall
               (10,10),# square
               (9,10)) # wide
    A = randn(m,n)
    @test svdvals(A) ≈ Vector{Float64}(svdvals(big(A)))
    @test cond(A) ≈ Float64(cond(big(A)))
end
