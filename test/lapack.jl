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
end
