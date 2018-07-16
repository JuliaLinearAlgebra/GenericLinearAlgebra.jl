using Test, GenericLinearAlgebra, LinearAlgebra, Quaternions

@testset "Singular value decomposition" begin
    @testset "Problem dimension ($m,$n)" for
        (m,n) in ((6,5)     , (6,6)     , (5,6),
                  (60, 50)  , (60, 60)  , (50, 60),
                  (200, 150), (200, 200), (150, 200))

        vals = reverse(collect(1:min(m,n)))
        U = qr(Quaternion{Float64}[Quaternion(randn(4)...) for i = 1:m, j = 1:min(m,n)]).Q
        V = qr(Quaternion{Float64}[Quaternion(randn(4)...) for i = 1:min(m,n), j = 1:n]).Q

        # FixMe! Using Array here shouldn't be necessary. Can be removed once
        # the bug in LinearAlgebra is fixed
        A = U*Array(Diagonal(vals))*V'

        @test vals ≈ svdvals(A)
    end

    @testset "The Ivan Slapničar Challenge" begin
        # This matrix used to hang (for n = 70). Thanks to Ivan Slapničar for reporting.
        n = 70
        J = Bidiagonal(0.5 * ones(n), ones(n-1), :U)
        @test GenericLinearAlgebra._svdvals!(copy(J)) ≈ svdvals(J)
        @test GenericLinearAlgebra._svdvals!(copy(J))[end] / svdvals(J)[end] - 1 < n*eps()
    end

    @testset "Extending Base methods. Problem dimension" for
        (m, n) in ((10,  9), # tall
                   (10, 10), # square
                   (9 , 10)) # wide

        A = randn(m,n)
        @test svdvals(A) ≈ Vector{Float64}(svdvals(big.(A)))
        @test cond(A)    ≈ Float64(cond(big.(A)))
    end
end
