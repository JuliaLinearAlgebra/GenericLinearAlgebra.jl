using Test, GenericLinearAlgebra, LinearAlgebra

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

@testset "Convergence in corner cases. Issue 29." begin
    function H(n::Int)
        H = zeros(2n, 2n)
        for i = 1 : 2 : 2n
            H[i, i+1] = 1
            H[i+1, i] = 1
        end
        H
    end

    function E(n::Int)
        E = zeros(2n, 2n)
        for i = 1 : (n - 1)
            E[2i + 1, 2i] = 1
        end
        E[1, 2n] = 1
        E
    end

    my_matrix(n::Int, η::Float64 = 1e-9) = H(n) .+ η .* E(n)

    A = my_matrix(4, 1e-3);
    @test sort(GenericLinearAlgebra._eigvals!(GenericLinearAlgebra._schur!(copy(A))), by = t -> (real(t), imag(t))) ≈ sort(eigvals(A), by = t -> (real(t), imag(t)))
end

end