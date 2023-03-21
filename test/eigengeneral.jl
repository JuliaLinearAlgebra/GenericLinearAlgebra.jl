using Test, GenericLinearAlgebra, LinearAlgebra

@testset "The General eigenvalue problem" begin

    cplxord = t -> (real(t), imag(t))

    @testset "General eigen problem with n=$n and element type=$T" for n in (10, 23, 100),
        T in (Float64, Complex{Float64})

        A = randn(T, n, n)
        vGLA = GenericLinearAlgebra._eigvals!(copy(A))
        vLAPACK = eigvals(A)
        vBig = eigvals(big.(A)) # not defined in LinearAlgebra so will dispatch to the version in GenericLinearAlgebra
        @test sort(vGLA, by = cplxord) ≈ sort(vLAPACK, by = cplxord)
        @test sort(vGLA, by = cplxord) ≈ sort(complex(eltype(A)).(vBig), by = cplxord)
        @test issorted(vBig, by = cplxord)

        if T <: Complex
            @testset "Rayleigh shifts" begin
                @test sort(
                    GenericLinearAlgebra._eigvals!(
                        GenericLinearAlgebra._schur!(copy(A), shiftmethod = :Rayleigh),
                    ),
                    by = t -> (real(t), imag(t)),
                ) ≈ sort(eigvals(A), by = t -> (real(t), imag(t)))
            end
        end
    end

    @testset "make sure that solver doesn't hang" begin
        for i = 1:1000
            A = randn(8, 8)
            sort(abs.(GenericLinearAlgebra._eigvals!(copy(A)))) ≈ sort(abs.(eigvals(A)))
        end
    end

    @testset "Convergence in corner cases. Issue 29." begin
        function H(n::Int)
            H = zeros(2n, 2n)
            for i = 1:2:2n
                H[i, i+1] = 1
                H[i+1, i] = 1
            end
            H
        end

        function E(n::Int)
            E = zeros(2n, 2n)
            for i = 1:(n-1)
                E[2i+1, 2i] = 1
            end
            E[1, 2n] = 1
            E
        end

        my_matrix(n::Int, η::Float64 = 1e-9) = H(n) .+ η .* E(n)

        A = my_matrix(4, 1e-3)
        @test sort(
            GenericLinearAlgebra._eigvals!(GenericLinearAlgebra._schur!(copy(A))),
            by = t -> (real(t), imag(t)),
        ) ≈ sort(eigvals(A), by = t -> (real(t), imag(t)))
    end

    @testset "Convergence in with 0s Issue 58." begin
        A = [0.0 1.0 0.0; -1.0 0.0 0.0; 0.0 0.0 0.0]
        @test sort(
            GenericLinearAlgebra._eigvals!(GenericLinearAlgebra._schur!(copy(A))),
            by = t -> (real(t), imag(t)),
        ) ≈ sort(eigvals(A), by = t -> (real(t), imag(t)))
        B = [0.0 0.0 0.0; 0.0 0.0 1.0; 0.0 -1.0 0.0]
        @test sort(
            GenericLinearAlgebra._eigvals!(GenericLinearAlgebra._schur!(copy(B))),
            by = t -> (real(t), imag(t)),
        ) ≈ sort(eigvals(B), by = t -> (real(t), imag(t)))
    end

    @testset "Extract Schur factor" begin
        A = randn(5, 5)
        @test sum(eigvals(schur(A).T)) ≈ sum(eigvals(Float64.(schur(big.(A)).T)))
        @test sum(eigvals(schur(A).Schur)) ≈ sum(eigvals(Float64.(schur(big.(A)).Schur)))
    end

    @testset "Issue 63" begin
        A = [
            1 2 1 1 1 1
            0 1 0 1 0 1
            1 1 2 0 1 0
            0 1 0 2 1 1
            1 1 1 0 2 0
            0 1 1 1 0 2
        ]

        z1 = [complex(1, sqrt(big(3))), complex(1, -sqrt(big(3)))]
        z2 = [
            4 * 2^(big(2) / 3) / complex(5, 3 * sqrt(big(111)))^(big(1) / 3),
            complex(5, 3 * sqrt(big(111)))^(big(1) / 3) / 2^(big(2) / 3),
        ]

        truevals =
            real.([
                (7 - transpose(z1) * z2),
                3,
                3,
                3,
                (7 - adjoint(z1) * z2),
                (7 + [2, 2]' * z2),
            ]) / 3

        @test eigvals(big.(A)) ≈ truevals
    end

    Demmel(η) = [
        0 1 0 0
        1 0 η 0
        0 -η 0 1
        0 0 1 0
    ]

    @testset "Demmel matrix" for t in (1e-10, 1e-9, 1e-8)
        # See "Sandia technical report 96-0913J: How the QR algorithm fails to converge and how fix it"
        A = Demmel(t)
        vals = GenericLinearAlgebra._eigvals!(GenericLinearAlgebra._schur!(A, maxiter = 35))
        @test abs.(vals) ≈ ones(4)
    end

    function Hevil2(θ, κ, α, γ)
        # Eq (13) and (14)
        β = ω = 0.0
        ν = cos(θ) * cos(2γ) + cos(α + β + ω) * sin(2γ) * κ / 2
        σ = 1 + κ * sin(2γ) * cos(α + β + ω - θ) + κ^2 * sin(γ)^2
        μ = -sin(θ) * cos(2γ) - sin(α + β + ω) * sin(2γ) * κ / 2
        ρ = sqrt(σ - ν^2)

        return [
            ν (cos(2θ)-ν^2)/ρ μ/ρ*(cos(2θ)-ν^2+ρ^2)/sqrt(ρ^2 - μ^2) (-2*μ*ν-sin(2 * θ))/sqrt(ρ^2 - μ^2)
            ρ -ν-sin(2 * θ)*μ/ρ^2 -μ/ρ^2*(μ*sin(2 * θ)+2*ν*ρ^2)/sqrt(ρ^2 - μ^2) -μ/ρ*(cos(2 * θ)-ν^2+ρ^2)/sqrt(ρ^2 - μ^2)
            0 sin(2 * θ)*sqrt(ρ^2 - μ^2)/ρ^2 ν+sin(2 * θ)*μ/ρ^2 (cos(2 * θ)-ν^2)/ρ
            0 0 ρ -ν
        ]
    end

    @testset "Complicate matrix from Sandia technical report" begin
        H = Hevil2(
            0.111866322512629152,
            1.08867072154101741,
            0.338146383137297168,
            -0.313987810419091240,
        )

        @test Float64.(abs.(eigvals(big.(H)))) ≈ ones(4)
    end

    @testset "Issue 67" for (A, λs) in (
        (
            [
                1 -2 1 -1 -1 0
                0 1 0 1 0 1
                1 -1 2 0 -1 0
                0 1 0 2 1 1
                1 0 1 0 0 0
                0 -1 1 -1 -2 0
            ],
            [
                0.0,
                0.0,
                (3 - √big(3) * im) / 2,
                (3 + √big(3) * im) / 2,
                (3 - √big(3)im) / 2,
                (3 + √big(3)im) / 2,
            ],
        ),
        (
            [
                1 0 -1 0 0 0
                0 1 1 1 -1 0
                1 0 -1 0 0 0
                1 -1 0 -1 1 1
                0 0 1 0 0 0
                -1 0 1 0 0 0
            ],
            zeros(6),
        ),
        (
            [
                1 -2 -1 1 -1 0
                0 1 0 -1 0 1
                -1 1 2 0 1 0
                0 -1 -2 2 -1 -1
                1 0 -1 0 0 0
                0 -1 -1 1 0 0
            ],
            [
                (1 - √big(3) * im) / 2,
                (1 + √big(3) * im) / 2,
                (1 - √big(3)im) / 2,
                (1 + √big(3)im) / 2,
                2,
                2,
            ],
        ),
        (
            [
                2 0 -1 0 0 1
                0 2 0 -1 -1 0
                -1 0 1 0 0 -1
                0 -1 0 1 1 0
                0 1 0 -1 0 0
                -1 0 1 0 0 0
            ],
            ones(6),
        ),
        (
            [
                1 0 1 0 -1 0
                -2 1 2 -1 1 1
                -1 0 -1 0 1 0
                2 1 2 -1 -2 1
                1 0 1 0 -1 0
                1 -1 -2 1 0 -1
            ],
            [-1, -1, 0, 0, 0, 0],
        ),
    )

        @testset "shouldn't need excessive iterations (30*n) in the Float64 case" begin
            GenericLinearAlgebra._schur!(float(A))
        end

        # For BigFloats, many iterations are required for convergence
        # Improving this is probably a hard problem
        vals = eigvals(big.(A), maxiter = 1500)

        # It's hard to compare complex conjugate pairs so we compare the real and imaginary parts separately
        @test sort(real(vals)) ≈ sort(real(λs)) atol = 1e-25
        @test sort(imag(vals)) ≈ sort(imag(λs)) atol = 1e-25
    end

    @testset "_hessenberg! and Hessenberg" begin
        n = 10
        A = randn(n, n)
        HF = GenericLinearAlgebra._hessenberg!(copy(A))
        for i = 1:length(HF.τ)
            HM = convert(Matrix, HF.τ[i])
            A[(i+1):end, :] = HM * A[(i+1):end, :]
            A[:, (i+1):end] = A[:, (i+1):end] * HM'
        end
        @test tril(A, -2) ≈ zeros(n, n) atol = 1e-14

        @test eigvals(HF.H) ≈ eigvals(A)
        @test eigvals(HF.H) ≈ eigvals!(copy(HF))
        @test HF.H \ ones(n) ≈ Matrix(HF.H) \ ones(n)
    end
end
