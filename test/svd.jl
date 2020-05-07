using Test, GenericLinearAlgebra, LinearAlgebra, Quaternions

@testset "Singular value decomposition" begin
    @testset "Problem dimension ($m,$n)" for
        (m,n) in ((6,5)     , (6,6)     , (5,6),
                  (60, 50)  , (60, 60)  , (50, 60),
                  (200, 150), (200, 200), (150, 200))

        vals = reverse(collect(1:min(m,n)))
        U = qr(Quaternion{Float64}[Quaternion(randn(4)...) for i = 1:m, j = 1:min(m,n)]).Q
        V = qr(Quaternion{Float64}[Quaternion(randn(4)...) for i = 1:n, j = 1:min(m,n)]).Q

        # FixMe! Using Array here shouldn't be necessary. Can be removed once
        # the bug in LinearAlgebra is fixed
        A = U*Array(Diagonal(vals))*V'

        @test size(A) == (m, n)
        @test vals ≈ svdvals(A)

        F = svd(A)
        @test vals ≈ F.S
        @test F.U'*A*F.V ≈ Diagonal(F.S)
    end

    @testset "The Ivan Slapničar Challenge" begin
        # This matrix used to hang (for n = 70). Thanks to Ivan Slapničar for reporting.
        n = 70
        J = Bidiagonal(0.5 * ones(n), ones(n-1), :U)
        @test GenericLinearAlgebra._svdvals!(copy(J)) ≈ svdvals(J)
        @test GenericLinearAlgebra._svdvals!(copy(J))[end] / svdvals(J)[end] - 1 < n*eps()
    end

    @testset "Compare to Base methods. Problem dimension ($m,$n)" for
        (m, n) in ((10,  9), # tall
                   (10, 10), # square
                   (9 , 10)) # wide

        A    = randn(m,n)
        Abig = big.(A)
        @test svdvals(A) ≈ Vector{Float64}(svdvals(Abig))
        @test cond(A)    ≈ Float64(cond(Abig))

        F    = svd(A)
        Fbig = svd(Abig)
        @test abs.(F.U'Float64.(Fbig.U)) ≈ I
        @test abs.(F.V'Float64.(Fbig.V)) ≈ I
    end

    @testset "Issue 54" begin
        U0, _, V0 = svd(big.(reshape(0:15, 4, 4)))
        A = U0[:, 1:3] * V0[:, 1:3]'

        U, S, V = svd(A)
        @test A ≈ U*Diagonal(S)*V'
    end
end
