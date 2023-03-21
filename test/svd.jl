using Test, GenericLinearAlgebra, LinearAlgebra, Quaternions, DoubleFloats

@testset "Singular value decomposition" begin
    @testset "Problem dimension ($m,$n)" for (m, n) in (
        (6, 5),
        (6, 6),
        (5, 6),
        (60, 50),
        (60, 60),
        (50, 60),
        (200, 150),
        (200, 200),
        (150, 200),
    )

        vals = reverse(collect(1:min(m, n)))
        U = qr(Quaternion{Float64}[Quaternion(randn(4)...) for i = 1:m, j = 1:min(m, n)]).Q
        V = qr(Quaternion{Float64}[Quaternion(randn(4)...) for i = 1:n, j = 1:min(m, n)]).Q

        # FixMe! Using Array here shouldn't be necessary. Can be removed once
        # the bug in LinearAlgebra is fixed
        A = U * Array(Diagonal(vals)) * V'

        @test size(A) == (m, n)
        @test vals ≈ svdvals(A)

        F = svd(A)
        @test vals ≈ F.S
        @show norm(F.U' * A * F.V - Diagonal(F.S), Inf)
        @test F.U' * A * F.V ≈ Diagonal(F.S)
    end

    @testset "The Ivan Slapničar Challenge" begin
        # This matrix used to hang (for n = 70). Thanks to Ivan Slapničar for reporting.
        n = 70
        J = Bidiagonal(0.5 * ones(n), ones(n - 1), :U)
        @test GenericLinearAlgebra._svdvals!(copy(J)) ≈ svdvals(J)
        @test GenericLinearAlgebra._svdvals!(copy(J))[end] / svdvals(J)[end] - 1 < n * eps()
    end

    @testset "Compare to Base methods. Problem dimension ($m,$n)" for (m, n) in (
        (10, 9), # tall
        (10, 10), # square
        (9, 10),
    ) # wide

        A = randn(m, n)
        Abig = big.(A)
        @test svdvals(A) ≈ Vector{Float64}(svdvals(Abig))
        @test cond(A) ≈ Float64(cond(Abig))

        F = svd(A)
        Fbig = svd(Abig)
        @test abs.(F.U'Float64.(Fbig.U)) ≈ I
        @test abs.(F.V'Float64.(Fbig.V)) ≈ I

        F = svd(A, full = true)
        Fbig = svd(Abig, full = true)
        @test abs.(F.U'Float64.(Fbig.U)) ≈ I
        @test abs.(F.V'Float64.(Fbig.V)) ≈ I
    end

    @testset "Issue 54" begin
        U0, _, V0 = svd(big.(reshape(0:15, 4, 4)))
        A = U0[:, 1:3] * V0[:, 1:3]'

        U, S, V = svd(A)
        @test A ≈ U * Diagonal(S) * V'
    end

    @testset "Very small matrices. Issue 79" begin
        A = randn(1, 2)
        FA = svd(A)
        FAb = svd(big.(A))
        FAtb = svd(big.(A'))
        @test FA.S ≈ Float64.(FAb.S) ≈ Float64.(FAtb.S)
        @test abs.(FA.U' * Float64.(FAb.U)) ≈ I
        @test abs.(FA.U' * Float64.(FAtb.V)) ≈ I
        @test abs.(FA.V' * Float64.(FAb.V)) ≈ I
        @test abs.(FA.V' * Float64.(FAtb.U)) ≈ I
    end

    @testset "Issue 81" begin
        A = [1 0 0 0; 0 2 1 0; 0 1 2 0; 0 0 0 -1]
        @test Float64.(svdvals(big.(A))) ≈ svdvals(A)

        A = [
            0.3 0.0 0.0 0.0 0.0 0.2 0.3 0.0
            0.0 0.0 0.0 0.0 0.1 0.0 0.0 0.0
            0.0 -0.2 0.0 0.0 0.0 0.0 0.0 -0.2
            0.3 0.0 0.0 0.0 0.0 0.2 0.4 0.0
            0.0 0.4 -0.2 0.0 0.0 0.0 0.0 0.3
            0.2 0.0 0.0 0.0 0.0 0.0 0.2 0.0
            0.0 0.0 0.0 0.1 0.0 0.0 0.0 0.0
            0.0 0.3 -0.2 0.0 0.0 0.0 0.0 0.3
        ]
        @test GenericLinearAlgebra._svdvals!(
            GenericLinearAlgebra.bidiagonalize!(copy(A)).bidiagonal,
        ) ≈ svdvals(A)

        n = 17
        A = zeros(Double64, n, n)
        for j = 1:n, i = 1:n
            A[i, j] = 1 / Double64(i + j - 1)
        end
        @test svdvals(A) ≈ svdvals(Float64.(A))

        # From https://github.com/JuliaMath/DoubleFloats.jl/issues/149
        n = 64
        c = Complex{BigFloat}(3 // 1 + 1im // 1)
        A = diagm(
            1 => c * ones(BigFloat, n - 1),
            -1 => c * ones(BigFloat, n - 1),
            -2 => ones(BigFloat, n - 2),
        )
        @test svdvals(A) ≈ svdvals(Complex{Double64}.(A))
    end

    @testset "Issue 104. Trailing zero in bidiagonal." begin
        dv = [
            -1.8066303423244812,
            0.23626846066361504,
            -1.8244461746384022,
            0.3743075843671794,
            -1.503025651470883,
            -0.5273978245088017,
            6.194053744695789e-75,
            -1.4816465601202412e-77,
            -7.05967042009753e-78,
            -1.8409609384104132e-78,
            -3.5760859484965067e-78,
            1.7012650564461077e-153,
            -5.106470534144341e-155,
            3.6429789846941095e-155,
            -3.494481025232055e-232,
            0.0,
        ]
        ev = [
            2.6390728646133144,
            1.9554155623906322,
            1.9171721320115487,
            2.5486042731357257,
            1.6188084135207441,
            -1.2764293576778472,
            -3.0873284294725004e-77,
            1.0815807869027443e-77,
            -1.0375522364338647e-77,
            -9.118279619242446e-78,
            5.910901980416107e-78,
            -7.522759136373737e-155,
            1.1750163871116314e-154,
            -2.169544740239464e-155,
            2.3352910098001318e-232,
        ]
        B = Bidiagonal(dv, ev, :U)
        F = GenericLinearAlgebra._svd!(copy(B))
        @test diag(F.U' * B * F.Vt') ≈ F.S rtol = 5e-15

        dv = [
            -2.130128478463753,
            1.1531468092039445,
            -2.847352842043162,
            0.6231779022541623,
            -1.690093069308479,
            -1.1624695467913961,
            -2.6642152887216867e-14,
            -0.33250510974785286,
            -2.2586294525518498,
            -1.1141061997716504,
            -3.1527435586000423,
            -1.823897129321229,
            -0.36124507542200524,
            -1.077527351579055e-14,
            -0.5940127483326197,
            -3.250185182454264,
            1.3384585640782973,
            3.5078636357910748e-16,
            -3.6405198891576764e-16,
            1.4444960997803543e-16,
            -2.7341646772961597e-16,
            -1.4813555234291996e-16,
            1.975443684819806e-16,
            1.6250251794596737e-16,
            -1.2910786368068701e-16,
            4.4994314760009794e-17,
            4.746123492548763e-17,
            -8.50145831025358e-17,
            -2.7086126595918776e-29,
            1.4778381126941581e-31,
            8.563148986816008e-33,
            -2.576078852735729e-33,
            -5.670281645899369e-33,
            -4.3644672970128596e-46,
            3.038140544755097e-48,
            0.0,
        ]
        ev = [
            2.1119692536821772,
            1.6448724864758517,
            -2.9887577094992634,
            -2.806523896769826,
            0.9686419967280151,
            -1.2565901258480143,
            -3.154707543364588,
            -2.128054620058084,
            1.263977392456184,
            7.354767862969798e-15,
            -0.36808384394360333,
            -2.688934874241276,
            1.329995154812558,
            -3.065722852484613,
            -0.6498657781673701,
            -0.0512439655384504,
            -5.784793952948882e-14,
            -3.748937717520342e-16,
            -3.451207270993785e-16,
            2.0899684974141464e-16,
            -2.794129724370907e-16,
            -2.2565321108962225e-16,
            1.7575656432452187e-16,
            1.029754558145647e-16,
            -1.5753211255094817e-16,
            1.4236049613292231e-16,
            -1.26511866970657e-16,
            -8.86788043706418e-17,
            2.2436351171246806e-30,
            -1.5828202903652256e-32,
            -1.1811335201359479e-32,
            -7.40990609589365e-33,
            -4.2582071949325636e-33,
            -1.8291280657969477e-46,
            1.9436291711597153e-50,
        ]
        B = Bidiagonal(dv, ev, :U)
        F = GenericLinearAlgebra._svd!(copy(B))
        @test diag(F.U' * B * F.Vt') ≈ F.S rtol = 5e-15
    end

    @testset "Generic HessenbergQ multiplication" begin
        A = big.(randn(10, 10))
        BF = GenericLinearAlgebra.bidiagonalize!(copy(A))
        @test (BF.rightQ' * Matrix(I, size(A)...)) * BF.rightQ ≈ I
    end
end
