using Test, GenericLinearAlgebra, LinearAlgebra, Quaternions

@testset "The selfadjoint eigen problem" begin
    n = 50
    @testset "SymTridiagonal" begin
        # Should automatically dispatch to method defined in this
        # package since a BigFloat methods isn't defined in
        # LinearAlgebra
        T = SymTridiagonal(big.(randn(n)), big.(randn(n - 1)))
        vals, vecs = eigen(T)
        @test issorted(vals)
        @testset "default" begin
            @test (vecs' * T) * vecs ≈ Diagonal(vals)
            @test eigvals(T) ≈ vals
            @test vecs'vecs ≈ Matrix(I, n, n)
        end

        @testset "eigen2" begin
            vals2, vecs2 = GenericLinearAlgebra.eigen2(T)
            @test issorted(vals2)
            @test vals ≈ vals2
            @test vecs[[1, n], :] == vecs2
            @test vecs2 * vecs2' ≈ Matrix(I, 2, 2)
        end

        @testset "QR version (QL is default)" begin
            vals, vecs =
                GenericLinearAlgebra.eigQR!(copy(T), vectors = Matrix{eltype(T)}(I, n, n))
            @test issorted(vals)
            @test (vecs' * T) * vecs ≈ Diagonal(vals)
            @test eigvals(T) ≈ vals
            @test vecs'vecs ≈ Matrix(I, n, n)
        end
    end

    @testset "(full) Symmetric" for uplo in (:L, :U)
        A = Hermitian(big.(randn(n, n)), uplo)
        vals, vecs = eigen(A)
        @testset "default" begin
            @test vecs' * A * vecs ≈ diagm(0 => vals)
            @test eigvals(A) ≈ vals
            @test vecs'vecs ≈ Matrix(I, n, n)
            @test issorted(vals)
        end

        @testset "eigen2" begin
            vals2, vecs2 = GenericLinearAlgebra.eigen2(A)
            @test vals ≈ vals2
            @test vecs[[1, n], :] ≈ vecs2
            @test vecs2 * vecs2' ≈ Matrix(I, 2, 2)
            @test issorted(vals2)
        end
    end

    @testset "(full) Quaternion Hermitian using :$uplo" for uplo in (:L, :U)
        V = qr([Quaternion(randn(4)...) for i = 1:n, j = 1:n]).Q
        λ = 10 .^ range(-8, stop = 0, length = n)
        A = Hermitian(V * Diagonal(λ) * V' |> t -> (t + t') / 2, uplo)
        vals, vecs = eigen(A)
        @test issorted(vals)

        @testset "default" begin
            if uplo == :L # FixMe! Probably an conjugation is off somewhere. Don't have time to check now.
                @test_broken vecs' * A * vecs ≈ diagm(0 => vals)
            else
                @test vecs' * A * vecs ≈ diagm(0 => vals)
            end
            @test eigvals(A) ≈ vals
            @test vals ≈ λ rtol = 1e-13 * n
            @test vecs'vecs ≈ Matrix(I, n, n)
        end

        @testset "eigen2" begin
            vals2, vecs2 = GenericLinearAlgebra.eigen2(A)
            @test issorted(vals2)
            @test vals ≈ vals2
            @test vecs[[1, n], :] ≈ vecs2
            @test vecs2 * vecs2' ≈ Matrix(I, 2, 2)
        end
    end

    @testset "big Hermitian{<:Complex}" begin
        # This one used to cause an ambiguity error. See #35
        A = complex.(randn(4, 4), randn(4, 4))
        @test Float64.(eigen(Hermitian(big.(A))).values) ≈ eigen(Hermitian(copy(A))).values
        @test Float64.(eigvals(Hermitian(big.(A)))) ≈ eigvals(Hermitian(copy(A)))
    end

    @testset "generic Givens" begin
        x, y = randn(2)
        c, s, r = invoke(LinearAlgebra.givensAlgorithm, Tuple{Real,Real}, x, y)
        @test c * x + s * y ≈ r
        @test c * y ≈ s * x
    end

    @testset "out-of-bounds issue in 1x1 case" begin
        @test GenericLinearAlgebra._eigvals!(SymTridiagonal([1.0], Float64[])) == [1.0]
        @test GenericLinearAlgebra._eigen!(SymTridiagonal([1.0], Float64[])).values == [1.0]
        @test GenericLinearAlgebra._eigen!(SymTridiagonal([1.0], Float64[])).vectors ==
              fill(1.0, 1, 1)
    end

    # Issue #52
    @testset "convergence criterion when diagonal has zeros" begin
        M1 = Hermitian(zeros(3, 3))
        M1[1, 1] = 0.01
        M2 = Hermitian(zeros(3, 3))
        M2[3, 3] = 0.01
        @test eigvals(M1) == GenericLinearAlgebra._eigvals!(copy(M1))
        @test eigvals(M2) == GenericLinearAlgebra._eigvals!(copy(M2))
        @test eigen(M1).values == GenericLinearAlgebra._eigen!(copy(M1)).values
        @test eigen(M2).values == GenericLinearAlgebra._eigen!(copy(M2)).values
    end

    @testset "Sorting of `ishermitian(T)==true` matrices on pre-1.2" begin
        T = big.(randn(5, 5))
        T = T + T'
        @test issorted(eigvals(T))
    end

    @testset "issue 123" begin
        M = Hermitian([
            big"-0.4080898675832881399369478084264191594976530854542904557798567397269356887436951";
            big"-0.1032324294981949906363774065395184125237581835226155628209100984396171211818558";
            big"-1.0795157507124452910839896877334667387210301781514938067860918240771876343947";
            big"0.9172086645212876240254394768180975107502376572771647296150618931226550446699544";;
            big"-0.1032324294981949906363774065395184125237581835226155628209100984396171211818558";
            big"-0.9819956883377066621250198846550622559246996804965712336465013506629992739010227";
            big"0.1882735697944729855991976669864503854920622386133987141371224931350749728226066";
            big"-0.1599663084136352437739757607131301560774255778371317602542426234968564801904052";;
            big"-1.0795157507124452910839896877334667387210301781514938067860918240771876343947";
            big"0.1882735697944729855991976669864503854920622386133987141371224931350749728226066";
            big"0.9688026817149176598146701814747478080649943014810992426739997593840858865727305";
            big"-1.672789745967021000172452940954243617442140494364475046869527486458478435262502";;
            big"0.9172086645212876240254394768180975107502376572771647296150618931226550446699544";
            big"-0.1599663084136352437739757607131301560774255778371317602542426234968564801904052";
            big"-1.672789745967021000172452940954243617442140494364475046869527486458478435262502";
            big"0.4212828742060771422472975116067336073573584644697624467523583310058490760719874"
        ])
        F = eigen(M)
        @test M * F.vectors ≈ F.vectors * Diagonal(F.values)

        @testset "2x2 cases, d: $d, e: $e" for (d, e) in (
            ([1.0, 1.0], [0.0]),
            ([0.0, 0.0], [1.0]),
            ([1.0, 1.0], [1e-20]),
            ([1e-20, 1e-30], [0.5]),
            ([1e-20, 1e-30], [1e-10]),
            ([1e-20, 1e-30], [1e-40]),
            ([1e20, 1e30], [0.5]),
            ([1e20, 1e30], [1e10]),
            ([1e20, 1e30], [1e40])
        )
            V = Matrix{Float64}(I, 2, 2)
            c, s = GenericLinearAlgebra.eig2x2!(d, e, 1, V)
            @test hypot(c, s) ≈ 1
        end
    end

    @testset "#133" begin
        A = SymTridiagonal{BigFloat}(randn(5), randn(4))
        T = Tridiagonal(A)
        @test eigvals(A) == eigvals(T) == eigvals(A; sortby=LinearAlgebra.eigsortby) == eigvals(T; sortby=LinearAlgebra.eigsortby) ==  eigvals!(deepcopy(A); sortby=LinearAlgebra.eigsortby)
        @test eigen(A).values == eigen(T).values == eigen(A; sortby=LinearAlgebra.eigsortby).values == eigen(T; sortby=LinearAlgebra.eigsortby).values
        # compare abs to avoid sign issues
        @test abs.(eigen(A).vectors) == abs.(eigen(T).vectors) == abs.(eigen(A; sortby=LinearAlgebra.eigsortby).vectors) == abs.(eigen(T; sortby=LinearAlgebra.eigsortby).vectors)
    end
end
