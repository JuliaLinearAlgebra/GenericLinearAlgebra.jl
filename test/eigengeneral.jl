if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

using LinearAlgebra

@testset "General eigen problems" begin
    n = 10
    A = randn(n,n)
    v1 = LinearAlgebra.EigenGeneral.eigvals!(copy(A))
    v2 = eigvals(A)
    vBig = LinearAlgebra.EigenGeneral.eigvals!(big(A))
    @test sort(real(v1)) ≈ sort(real(v2))
    @test sort(imag(v1)) ≈ sort(imag(v2))
    @test sort(real(v1)) ≈ sort(real(map(Complex{Float64}, vBig)))
    @test sort(imag(v1)) ≈ sort(imag(map(Complex{Float64}, vBig)))
end