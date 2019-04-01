using Test
using LinearAlgebra
using GenericLinearAlgebra

@testset "large matrix inv  " begin
    @testset "Problem dimension ($m,$m) with block size $bz" for
        (m, bz) in ( # Standard
                    ( 100,  50), ( 1000,  50), (1000,  500), (10_000, 5_000),
                     # Nondivisable by block size
                    ( 100,  64), ( 1000,  151), (1000,  331), (10_000, 6331),
                   )

        A = rand(m, m)
        A_inv = large_inv(A; max_block_size=bz)
        @test A_inv*A - I ≈ zeros(m,m) atol=1e-4
    end

    @testset "Error paths" begin
        @test_throws DimensionMismatch large_inv(ones(700,300); max_block_size=100)
        @test_throws ArgumentError large_inv(ones(100,100); max_block_size=1000)
    end
end
