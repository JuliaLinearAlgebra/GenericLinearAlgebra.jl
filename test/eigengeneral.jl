using Test, GenericLinearAlgebra, LinearAlgebra

@testset "The General eigenvalue problem" begin

cplxord = t -> (real(t), imag(t))

@testset "General eigen problem with n=$n and element type=$T" for
    n in (10, 23, 100),
    T in (Float64, Complex{Float64})

    A = randn(T, n, n)
    vGLA    = GenericLinearAlgebra._eigvals!(copy(A))
    vLAPACK = eigvals(A)
    vBig    = eigvals(big.(A)) # not defined in LinearAlgebra so will dispatch to the version in GenericLinearAlgebra
    @test sort(vGLA, by = cplxord) ≈ sort(vLAPACK, by = cplxord)
    @test sort(vGLA, by = cplxord) ≈ sort(complex(eltype(A)).(vBig), by = cplxord)
    if VERSION > v"1.2.0-DEV.0"
        @test issorted(vBig, by = cplxord)
    end
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

@testset "Convergence in with 0s Issue 58." begin
    A = [0.0 1.0 0.0; -1.0 0.0 0.0; 0.0 0.0 0.0]
    @test sort(GenericLinearAlgebra._eigvals!(GenericLinearAlgebra._schur!(copy(A))), by = t -> (real(t), imag(t))) ≈ sort(eigvals(A), by = t -> (real(t), imag(t)))
    B = [0.0 0.0 0.0; 0.0 0.0 1.0; 0.0 -1.0 0.0]
    @test sort(GenericLinearAlgebra._eigvals!(GenericLinearAlgebra._schur!(copy(B))), by = t -> (real(t), imag(t))) ≈ sort(eigvals(B), by = t -> (real(t), imag(t)))
end

@testset "Extract Schur factor" begin
    A = randn(5, 5)
    @test sum(eigvals(schur(A).T))     ≈ sum(eigvals(Float64.(schur(big.(A)).T)))
    @test sum(eigvals(schur(A).Schur)) ≈ sum(eigvals(Float64.(schur(big.(A)).Schur)))
end

end
