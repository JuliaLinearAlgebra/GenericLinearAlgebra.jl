module HouseholderModule

    using ..JuliaBLAS: rankUpdate!
    using Base.LinAlg: BlasReal, axpy!

    import Base: *
    import Base: Ac_mul_B, convert, full, size
    import Base.LinAlg: A_mul_B!, Ac_mul_B!

    using Compat
    import Compat.view

    immutable Householder{T,S<:StridedVector}
        v::S
        τ::T
    end
    immutable HouseholderBlock{T,S<:StridedMatrix,U<:StridedMatrix}
        V::S
        T::UpperTriangular{T,U}
    end

    # see dlapy2.f in LAPACK
    function lapy{T<:Real}(x::T, y::T)
        xabs = abs(x)
        yabs = abs(y)
        w = max(xabs, yabs)
        z = min(xabs, yabs)
        z == 0 ? w : w*sqrt(one(T) + (z/w)^2)
    end
    # see dlapy3.f in LAPACK
    function lapy(z::Complex, x::Real)
        zrabs = abs(real(z))
        ziabs = abs(imag(z))
        xabs = abs(x)
        w = max(zrabs, ziabs, xabs)
        w == 0 ? zrabs + ziabs + xabs : w*sqrt((zrabs/w)^2 + (ziabs/w)^2 + (xabs/w)^2)
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

        V1 = LinAlg.UnitLowerTriangular(view(V, 1:blocksize, 1:blocksize))
        V2 = view(V, blocksize+1:mA, 1:blocksize)
        A1 = view(A, 1:blocksize, 1:nA)
        A2 = view(A, blocksize+1:mA, 1:nA)
        copy!(M, A1)
        Ac_mul_B!(V1, M)
        # M = V1'A1
        Ac_mul_B!(one(T), V2, A2, one(T), M)
        A_mul_B!(H.T, M)
        A_mul_B!(-one(T), V2, M, one(T), A2)
        A_mul_B!(-one(T), V1, M)
        axpy!(one(T), M, A1)
        A
    end
    (*){T}(H::HouseholderBlock{T}, A::StridedMatrix{T}) =
        A_mul_B!(H, copy(A), similar(A, (min(size(H.V)...), size(A, 2))))

    function Ac_mul_B!{T}(H::HouseholderBlock{T}, A::StridedMatrix{T}, M::StridedMatrix)
        V = H.V
        mA, nA = size(A)
        # nH, blocksize = size(V)
        nH = size(V, 1)
        blocksize = min(nH, size(V, 2))
        nH == mA || throw(DimensionMismatch(""))

        V1 = LinAlg.UnitLowerTriangular(view(V, 1:blocksize, 1:blocksize))
        V2 = view(V, blocksize+1:mA, 1:blocksize)
        A1 = view(A, 1:blocksize, 1:nA)
        A2 = view(A, blocksize+1:mA, 1:nA)
        copy!(M, A1)
        Ac_mul_B!(V1, M)
        # M = V1'A1
        Ac_mul_B!(one(T), V2, A2, one(T), M)
        Ac_mul_B!(H.T, M)
        A_mul_B!(-one(T), V2, M, one(T), A2)
        A_mul_B!(-one(T), V1, M)
        axpy!(one(T), M, A1)
        A
    end
    Ac_mul_B{T}(H::HouseholderBlock{T}, A::StridedMatrix{T}) =
        Ac_mul_B!(H, copy(A), similar(A, (min(size(H.V)...), size(A, 2))))

    convert{T}(::Type{Matrix}, H::Householder{T}) = A_mul_B!(H, eye(T, size(H, 1)))
    convert{T}(::Type{Matrix{T}}, H::Householder{T}) = A_mul_B!(H, eye(T, size(H, 1)))

    full(H::Householder) = convert(Matrix, H)
end