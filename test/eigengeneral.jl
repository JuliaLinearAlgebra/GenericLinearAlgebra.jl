using GenericLinearAlgebra

@testset "The General eigenvalue problem" begin

@testset "General eigen problem with n=$n" for n in (10, 100, 200)
    A = randn(n,n)
    vGLA    = GenericLinearAlgebra._eigvals!(copy(A))
    vLAPACK = eigvals(A)
    vBig    = eigvals(big.(A)) # not defined in LinearAlgebra so will dispatch to the version in GenericLinearAlgebra
    @test sort(real(vGLA)) ≈ sort(real(vLAPACK))
    @test sort(imag(vGLA)) ≈ sort(imag(vLAPACK))
    @test sort(real(vGLA)) ≈ sort(real(map(Complex{Float64}, vBig)))
    @test sort(imag(vGLA)) ≈ sort(imag(map(Complex{Float64}, vBig)))
end

@testset "make sure that solver doesn't hang" begin
    for i in 1:1000
        A = randn(8, 8)
        sort(abs.(GenericLinearAlgebra._eigvals!(copy(A)))) ≈ sort(abs.(eigvals(A)))
    end
end

end