module HouseholderModule

    using ..JuliaBLAS: rankUpdate!
    using LinearAlgebra: BlasReal, axpy!

    import Base: *
    import Base: convert, size
    import LinearAlgebra: Ac_mul_B, A_mul_B!, Ac_mul_B!

    immutable Householder{T,S<:StridedVector}
        v::S
        τ::T
    end
    immutable HouseholderBlock{T,S<:StridedMatrix,U<:StridedMatrix}
        V::S
        T::UpperTriangular{T,U}
    end

    size(H::Householder) = (length(H.v), length(H.v))
    size(H::Householder, i::Integer) = i <= 2 ? length(H.v) : 1

    function A_mul_B!(H::Householder, A::StridedMatrix)
        m, n = size(A)
        length(H.v) == m - 1 || throw(DimensionMismatch(""))
        v = view(H.v, 1:m - 1)
        τ = H.τ
        for j = 1:n
            va = A[1,j]
            Aj = view(A, 2:m, j)
            va += dot(v, Aj)
            va = τ*va
            A[1,j] -= va
            axpy!(-va, v, Aj)
        end
        A
    end

    function A_mul_B!(A::StridedMatrix, H::Householder)
        m, n = size(A)
        length(H.v) == n - 1 || throw(DimensionMismatch(""))
        v = view(H.v, :)
        τ = H.τ
        a1 = view(A, :, 1)
        A1 = view(A, :, 2:n)
        x = A1*v
        axpy!(one(τ), a1, x)
        axpy!(-τ, x, a1)
        rankUpdate!(-τ, x, v, A1)
        A
    end

    function Ac_mul_B!(H::Householder, A::StridedMatrix)
        m, n = size(A)
        length(H.v) == m - 1 || throw(DimensionMismatch(""))
        v = view(H.v, 1:m - 1)
        τ = H.τ
        for j = 1:n
            va = A[1,j]
            Aj = view(A, 2:m, j)
            va += dot(v, Aj)
            va = τ'va
            A[1,j] -= va
            axpy!(-va, v, Aj)
        end
        A
    end

    function A_mul_B!{T}(H::HouseholderBlock{T}, A::StridedMatrix{T}, M::StridedMatrix{T})
        V = H.V
        mA, nA = size(A)
        nH = size(V, 1)
        blocksize = min(nH, size(V, 2))
        nH == mA || throw(DimensionMismatch(""))

        # Reflector block is split into a UnitLowerTriangular top part and rectangular lower part
        V1 = LinAlg.UnitLowerTriangular(view(V, 1:blocksize, 1:blocksize))
        V2 = view(V, blocksize+1:mA, 1:blocksize)

        # Split A to match the split in the reflector block
        A1 = view(A, 1:blocksize, 1:nA)
        A2 = view(A, blocksize+1:mA, 1:nA)

        # Copy top part of A (A1) to temporary work space
        copy!(M, A1)
        # Multiply UnitLowerTriangular V1 and A1 in-place in M (M = V1'A1)
        Ac_mul_B!(V1, M)
        # Add V2'A2 to V1'A1 (M := M + V2'A2)
        Ac_mul_B!(one(T), V2, A2, one(T), M)
        # Multiply the elementary block loading T (UpperTriangular) and M in-place
        A_mul_B!(H.T, M)

        # A2 := A2 - V2*M
        A_mul_B!(-one(T), V2, M, one(T), A2)
        # A1 := A1 - V1*M but since V1 is UnitLowerTriangular we do it in two steps:
        ## 1. M  := -V1*M
        ## 2. A1 := A1 + M
        A_mul_B!(-one(T), V1, M)
        axpy!(one(T), M, A1)

        return A
    end
    (*){T}(H::HouseholderBlock{T}, A::StridedMatrix{T}) =
        A_mul_B!(H, copy(A), similar(A, (min(size(H.V)...), size(A, 2))))

    function Ac_mul_B!{T}(H::HouseholderBlock{T}, A::StridedMatrix{T}, M::StridedMatrix)
        V = H.V
        mA, nA = size(A)
        nH = size(V, 1)
        blocksize = min(nH, size(V, 2))
        nH == mA || throw(DimensionMismatch(""))

        # Reflector block is split into a UnitLowerTriangular top part and rectangular lower part
        V1 = LinAlg.UnitLowerTriangular(view(V, 1:blocksize, 1:blocksize))
        V2 = view(V, blocksize+1:mA, 1:blocksize)

        # Split A to match the split in the reflector block
        A1 = view(A, 1:blocksize, 1:nA)
        A2 = view(A, blocksize+1:mA, 1:nA)

        # Copy top part of A (A1) to temporary work space
        copy!(M, A1)
        # Multiply UnitLowerTriangular V1 and A1 in-place in M (M = V1'A1)
        Ac_mul_B!(V1, M)
        # Add V2'A2 to V1'A1 (M := M + V2'A2)
        Ac_mul_B!(one(T), V2, A2, one(T), M)
        # Multiply the elementary block loading T (UpperTriangular) and M in-place
        Ac_mul_B!(H.T, M)

        # A2 := A2 - V2*M
        A_mul_B!(-one(T), V2, M, one(T), A2)
        # A1 := A1 - V1*M but since V1 is UnitLowerTriangular we do it in two steps:
        ## 1. M  := -V1*M
        ## 2. A1 := A1 + M
        A_mul_B!(-one(T), V1, M)
        axpy!(one(T), M, A1)

        return A
    end
    Ac_mul_B{T}(H::HouseholderBlock{T}, A::StridedMatrix{T}) =
        Ac_mul_B!(H, copy(A), similar(A, (min(size(H.V)...), size(A, 2))))

    convert{T}(::Type{Matrix}, H::Householder{T}) = A_mul_B!(H, eye(T, size(H, 1)))
    convert{T}(::Type{Matrix{T}}, H::Householder{T}) = A_mul_B!(H, eye(T, size(H, 1)))
end
