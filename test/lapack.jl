using Test, GenericLinearAlgebra, LinearAlgebra
using GenericLinearAlgebra.LAPACK2

@testset "LAPACK wrappers" begin

    n = 100

    T = SymTridiagonal(randn(n), randn(n - 1))
    vals, vecs = eigen(T)
    @testset "steqr" begin
        _vals, _vecs = LAPACK2.steqr!('N', copy(T.dv), copy(T.ev))
        @test vals ≈ _vals

        _vals, _vecs = LAPACK2.steqr!('I', copy(T.dv), copy(T.ev))
        @test vals ≈ _vals
        @test abs.(_vecs'vecs) ≈ Matrix(I, n, n)

        _vals, _vecs = LAPACK2.steqr!('V', copy(T.dv), copy(T.ev), Matrix{Float64}(I, n, n))
        @test vals ≈ _vals
        @test abs.(_vecs'vecs) ≈ Matrix(I, n, n)
    end

    @testset "sterf" begin
        _vals = LAPACK2.sterf!(copy(T.dv), copy(T.ev))
        @test _vals ≈ vals
    end

    @testset "stedc" begin
        _vals, _vecs = LAPACK2.stedc!('N', copy(T.dv), copy(T.ev))
        @test vals ≈ _vals

        _vals, _vecs = LAPACK2.stedc!('I', copy(T.dv), copy(T.ev))
        @test vals ≈ _vals
        @test abs.(_vecs'vecs) ≈ Matrix(I, n, n)

        _vals, _vecs = LAPACK2.stedc!('V', copy(T.dv), copy(T.ev), Matrix{Float64}(I, n, n))
        @test vals ≈ _vals
        @test abs.(_vecs'vecs) ≈ Matrix(I, n, n)
    end

    @testset "stemr" begin
        _vals, _vecs, __ = LAPACK2.stemr!('N', 'A', copy(T.dv), copy(T.ev))
        @test vals ≈ _vals

        _vals, _vecs, __ = LAPACK2.stemr!('N', 'V', copy(T.dv), copy(T.ev))
        @test vals ≈ _vals

        _vals, _vecs, __ = LAPACK2.stemr!('N', 'I', copy(T.dv), copy(T.ev))
        @test vals ≈ _vals

        _vals, _vecs, __ = LAPACK2.stemr!('V', 'A', copy(T.dv), copy(T.ev))
        @test vals ≈ _vals
        @test abs.(_vecs'vecs) ≈ Matrix(I, n, n)

        _vals, _vecs, __ = LAPACK2.stemr!('V', 'V', copy(T.dv), copy(T.ev))
        @test vals ≈ _vals
        @test abs.(_vecs'vecs) ≈ Matrix(I, n, n)

        _vals, _vecs, __ = LAPACK2.stemr!('V', 'I', copy(T.dv), copy(T.ev))
        @test vals ≈ _vals
        @test abs.(_vecs'vecs) ≈ Matrix(I, n, n)
    end

    @testset "lahqr" begin
        T = Array(Tridiagonal(fill(0.1, 99), fill(0.0, 100), fill(40.0, 99)))
        _vals = sort(LAPACK2.lahqr!(copy(T))[1])
        @test _vals ≈ sort(real.(GenericLinearAlgebra._eigvals!(copy(T))))
        # LAPACK's multishift algorithm (the default) seems to be broken
        @test !(_vals ≈ sort(eigvals(T)))
    end

    @testset "syevd: eltype=$eltype, uplo=$uplo" for eltype in (Float32, Float64, ComplexF32, ComplexF64), uplo in ('U', 'L')
        A = randn(eltype, n, n)
        A = A + A'
        if eltype <: Real
            vals, vecs = LAPACK2.syevd!('V', uplo, copy(A))
        else
            vals, vecs = LAPACK2.heevd!('V', uplo, copy(A))
        end
        @test diag(vecs'*A*vecs) ≈ eigvals(A)
    end

    @testset "tgevc: eltype=$eltype, side=$side, howmny=$howmny" for eltype in (Float32, Float64), side in ('L', 'R', 'B'), howmny in ('A', #='B', =#'S')
        select = ones(Int, n)
        S, P = triu(randn(eltype, n, n)), triu(randn(eltype, n, n))
        VL, VR, m = LAPACK2.tgevc!(
            side,
            howmny,
            select,
            copy(S),
            copy(P),
        )
        if side ∈ ('R', 'B')
            w = diag(S*VR) ./ diag(P*VR)
            @test S*VR ≈ P*VR*Diagonal(w) rtol=sqrt(eps(eltype)) atol=sqrt(eps(eltype))
        end
        if side ∈ ('L', 'B')
            w = w = diag(VL'*S) ./ diag(VL'*P)
            @test VL'*S ≈ Diagonal(w)*VL'*P rtol=sqrt(eps(eltype)) atol=sqrt(eps(eltype))
        end
    end
end
