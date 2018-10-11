using Test, GenericLinearAlgebra, LinearAlgebra, Quaternions
Base.isreal(q::Quaternion) = q.v1 == q.v2 == q.v3 == 0

@testset "The selfadjoint eigen problem" begin
    n = 50
    @testset "SymTridiagonal" begin
        # Should automatically dispatch to method defined in this
        # package since a BigFloat methods isn't defined in
        # LinearAlgebra
        T = SymTridiagonal(big.(randn(n)), big.(randn(n - 1)))
        vals, vecs  = eigen(T)
        @testset "default" begin
            @test (vecs'*T)*vecs ≈ Diagonal(vals)
            @test eigvals(T) ≈ vals
            @test vecs'vecs ≈ Matrix(I, n, n)
        end

        @testset "eigen2" begin
            vals2, vecs2 = GenericLinearAlgebra.eigen2(T)
            @test vals ≈ vals2
            @test vecs[[1,n],:] == vecs2
            @test vecs2*vecs2' ≈ Matrix(I, 2, 2)
        end

        @testset "QR version (QL is default)" begin
            vals, vecs = GenericLinearAlgebra.eigQR!(copy(T), vectors = Matrix{eltype(T)}(I, n, n))
            @test (vecs'*T)*vecs ≈ Diagonal(vals)
            @test eigvals(T) ≈ vals
            @test vecs'vecs ≈ Matrix(I, n, n)
        end
    end

    @testset "(full) Symmetric" for uplo in (:L, :U)
        A = Hermitian(big.(randn(n, n)), uplo)
        vals, vecs = eigen(A)
        @testset "default" begin
            @test vecs'*A*vecs ≈ diagm(0 => vals)
            @test eigvals(A) ≈ vals
            @test vecs'vecs ≈ Matrix(I, n, n)
        end

        @testset "eigen2" begin
            vals2, vecs2 = GenericLinearAlgebra.eigen2(A)
            @test vals ≈ vals2
            @test vecs[[1,n],:] ≈ vecs2
            @test vecs2*vecs2'  ≈ Matrix(I, 2, 2)
        end
    end

    @testset "(full) Quaternion Hermitian using :$uplo" for uplo in (:L, :U)
        V = qr([Quaternion(randn(4)...) for i in 1:n, j in 1:n]).Q
        λ = 10 .^ range(-8, stop=0, length=n)
        A = Hermitian(V*Diagonal(λ)*V' |> t -> (t + t')/2, uplo)
        vals, vecs = eigen(A)
        @testset "default" begin
            if uplo == :L # FixMe! Probably an conjugation is off somewhere. Don't have time to check now.
                @test_broken vecs'*A*vecs ≈ diagm(0=> vals)
            else
                @test vecs'*A*vecs ≈ diagm(0 => vals)
            end
            @test eigvals(A)   ≈ vals
            @test vals         ≈ λ rtol=1e-13*n
            @test vecs'vecs    ≈ Matrix(I, n, n)
        end

        @testset "eigen2" begin
            vals2, vecs2 = GenericLinearAlgebra.eigen2(A)
            @test vals ≈ vals2
            @test vecs[[1,n],:] ≈ vecs2
            @test vecs2*vecs2'  ≈ Matrix(I, 2, 2)
        end
    end

    @testset "big Hermitian{<:Complex}" begin
        # This one used to cause an ambiguity error. See #35
        A = complex.(randn(4,4), randn(4,4))
        @test Float64.(eigen(Hermitian(big.(A))).values) ≈ eigen(Hermitian(copy(A))).values
        @test Float64.(eigvals(Hermitian(big.(A))))      ≈ eigvals(Hermitian(copy(A)))
    end

    @testset "generic Givens" begin
        x, y = randn(2)
        c, s, r = invoke(LinearAlgebra.givensAlgorithm, Tuple{Real,Real}, x, y)
        @test c*x + s*y ≈ r
        @test c*y ≈ s*x
    end

    @testset "out-of-bounds issue in 1x1 case" begin
        @test GenericLinearAlgebra._eigvals!(SymTridiagonal([1.0], Float64[])) == [1.0]
        @test GenericLinearAlgebra._eigen!(SymTridiagonal([1.0], Float64[])).values == [1.0]
        @test GenericLinearAlgebra._eigen!(SymTridiagonal([1.0], Float64[])).vectors == fill(1.0, 1, 1)
    end
end
