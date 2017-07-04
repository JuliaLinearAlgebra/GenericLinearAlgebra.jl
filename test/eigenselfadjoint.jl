using Base.Test, LinearAlgebra

@testset "The selfadjoint eigen problem" begin
    n = 200
    @testset "SymTridiagonal" begin
        T = SymTridiagonal(randn(n), randn(n - 1))
        vals, vecs  = LinearAlgebra.EigenSelfAdjoint.eig(T)
        @testset "default" begin
            @test (vecs'*T)*vecs ≈ Diagonal(vals)
            @test LinearAlgebra.EigenSelfAdjoint.eigvals(T) ≈ vals
            @test vecs'vecs ≈ eye(n)
        end

        @testset "eig2" begin
            vals2, vecs2 = LinearAlgebra.EigenSelfAdjoint.eig2(T)
            @test vals ≈ vals2
            @test vecs[[1,n],:] == vecs2
            @test vecs2*vecs2' ≈ eye(2)
        end

        @testset "QR version (QL is default)" begin
            vals, vecs = LinearAlgebra.EigenSelfAdjoint.eigQR!(copy(T), eye(n))
            @test (vecs'*T)*vecs ≈ Diagonal(vals)
            @test LinearAlgebra.EigenSelfAdjoint.eigvals(T) ≈ vals
            @test vecs'vecs ≈ eye(n)
        end
    end

    @testset "(full) Symmetric" begin
        A = Symmetric(randn(n, n), :L)
        vals, vecs = LinearAlgebra.EigenSelfAdjoint.eig(A)
        @testset "default" begin
            @test vecs'*A*vecs ≈ diagm(vals)
            @test LinearAlgebra.EigenSelfAdjoint.eigvals(A) ≈ vals
            @test vecs'vecs ≈ eye(n)
        end

        @testset "eig2" begin
            vals2, vecs2 = LinearAlgebra.EigenSelfAdjoint.eig2(A)
            @test vals ≈ vals2
            @test vecs[[1,n],:] ≈ vecs2
            @test vecs2*vecs2'  ≈ eye(2)
        end
    end

    @testset "generic Givens" begin
        x, y = randn(2)
        c, s, r = invoke(LinAlg.givensAlgorithm, Tuple{Real,Real}, x, y)
        @test c*x + s*y ≈ r
        @test c*y ≈ s*x
    end
end
