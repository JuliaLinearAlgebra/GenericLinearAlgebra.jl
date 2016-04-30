using Base.Test
using LinearAlgebra

n = 10
A = randn(n,n)
v1 = LinearAlgebra.EigenGeneral.eigvals!(copy(A))
v2 = eigvals(A)
vBig = LinearAlgebra.EigenGeneral.eigvals!(big(A))
@test sort(real(v1)) ≈ sort(real(v2))
@test sort(imag(v1)) ≈ sort(imag(v2))
@test sort(real(v1)) ≈ sort(real(map(Complex{Float64}, vBig)))
@test sort(imag(v1)) ≈ sort(imag(map(Complex{Float64}, vBig)))
