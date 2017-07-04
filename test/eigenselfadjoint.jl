using Base.Test, LinearAlgebra

@testset "The selfadjoint eigen problem" begin
    n = 200
    @testset "SymTridiagonal" begin
        T = SymTridiagonal(randn(n), randn(n - 1))
        vals, vecs  = LinearAlgebra.EigenSelfAdjoint.eig(T)
        @test (vecs'*T)*vecs ≈ Diagonal(vals)
        @test eigvals(T) ≈ vals
        @test vecs'vecs ≈ eye(n)

        # test eig2
        vals2, vecs2 = LinearAlgebra.EigenSelfAdjoint.eig2(T)
        @test vals2 ≈ vals
        @test vecs[[1,end],:] == vecs2
        @test eigvals(T) ≈ vals2
        @test vecs2*vecs2' ≈ eye(2)

        # Test QR version (QL is default)
        valsQR, vecsQR = LinearAlgebra.EigenSelfAdjoint.eigQR!(copy(T), eye(n))
        @test valsQR ≈ vals
        @test abs.(vecsQR'vecs) ≈ eye(n)
    end

    @testset "(full) Symmetric" begin
        A = Symmetric(randn(n, n), :L)
        vals, vecs = LinearAlgebra.EigenSelfAdjoint.eig(A)
        @test vecs'*A*vecs ≈ diagm(vals)
        @test eigvals(A) ≈ vals
    end
end
