using GenericLinearAlgebra

@testset "General eigen problems" begin
    n = 10
    A = randn(n,n)
    v1 = LinearAlgebra.eigvals!(copy(A))
    v2 = eigvals(A)
    vBig = LinearAlgebra.eigvals!(big.(A))
    @test sort(real(v1)) ≈ sort(real(v2))
    @test sort(imag(v1)) ≈ sort(imag(v2))
    @test sort(real(v1)) ≈ sort(real(map(Complex{Float64}, vBig)))
    @test sort(imag(v1)) ≈ sort(imag(map(Complex{Float64}, vBig)))
end

@testset "make sure that solver doesn't hang" begin
    for i in 1:1000
        A = randn(8, 8)
        sort(abs.(LinearAlgebra.eigvals!(copy(A)))) ≈ sort(abs.(eigvals(A)))
    end
end