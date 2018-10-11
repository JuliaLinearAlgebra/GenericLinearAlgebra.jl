import Base: *, eltype, size
import LinearAlgebra: adjoint, mul!, rmul!

struct Householder{T,S<:StridedVector}
    v::S
    τ::T
end
struct HouseholderBlock{T,S<:StridedMatrix,U<:StridedMatrix}
    V::S
    T::UpperTriangular{T,U}
end

size(H::Householder) = (length(H.v), length(H.v))
size(H::Householder, i::Integer) = i <= 2 ? length(H.v) : 1

eltype(H::Householder{T})      where T = T
eltype(H::HouseholderBlock{T}) where T = T

adjoint(H::Householder{T})      where {T} = Adjoint{T,typeof(H)}(H)
adjoint(H::HouseholderBlock{T}) where {T} = Adjoint{T,typeof(H)}(H)

function lmul!(H::Householder, A::StridedMatrix)
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

function rmul!(A::StridedMatrix, H::Householder)
    m, n = size(A)
    length(H.v) == n - 1 || throw(DimensionMismatch(""))
    v = view(H.v, :)
    τ = H.τ
    a1 = view(A, :, 1)
    A1 = view(A, :, 2:n)
    x = A1*v
    axpy!(one(τ), a1, x)
    axpy!(-τ, x, a1)
    rankUpdate!(A1, x, v, -τ)
    A
end

function lmul!(adjH::Adjoint{<:Any,<:Householder}, A::StridedMatrix)
    H = parent(adjH)
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

# FixMe! This is a weird multiplication method and at least its behavior needs to be explained
function lmul!(H::HouseholderBlock{T}, A::StridedMatrix{T}, M::StridedMatrix{T}) where T
    V = H.V
    mA, nA = size(A)
    nH = size(V, 1)
    blocksize = min(nH, size(V, 2))
    nH == mA || throw(DimensionMismatch(""))

    # Reflector block is split into a UnitLowerTriangular top part and rectangular lower part
    V1 = LinearAlgebra.UnitLowerTriangular(view(V, 1:blocksize, 1:blocksize))
    V2 = view(V, blocksize+1:mA, 1:blocksize)

    # Split A to match the split in the reflector block
    A1 = view(A, 1:blocksize, 1:nA)
    A2 = view(A, blocksize+1:mA, 1:nA)

    # Copy top part of A (A1) to temporary work space
    copyto!(M, A1)
    # Multiply UnitLowerTriangular V1 and A1 in-place in M (M = V1'A1)
    lmul!(V1', M)
    # Add V2'A2 to V1'A1 (M := M + V2'A2)
    mul!(M, V2', A2, one(T), one(T))
    # Multiply the elementary block loading T (UpperTriangular) and M in-place
    lmul!(H.T, M)

    # A2 := A2 - V2*M
    mul!(A2, V2, M, -one(T), one(T))
    # A1 := A1 - V1*M but since V1 is UnitLowerTriangular we do it in two steps:
    ## 1. M  := -V1*M
    ## 2. A1 := A1 + M
    lmul!(V1, M, -one(T))
    axpy!(one(T), M, A1)

    return A
end
(*)(H::HouseholderBlock{T}, A::StridedMatrix{T}) where {T} =
        lmul!(H, copy(A), similar(A, (min(size(H.V)...), size(A, 2))))

function lmul!(adjH::Adjoint{T,<:HouseholderBlock{T}}, A::StridedMatrix{T}, M::StridedMatrix) where T
    H = parent(adjH)
    V = H.V
    mA, nA = size(A)
    nH = size(V, 1)
    blocksize = min(nH, size(V, 2))
    nH == mA || throw(DimensionMismatch(""))

    # Reflector block is split into a UnitLowerTriangular top part and rectangular lower part
    V1 = LinearAlgebra.UnitLowerTriangular(view(V, 1:blocksize, 1:blocksize))
    V2 = view(V, blocksize+1:mA, 1:blocksize)

    # Split A to match the split in the reflector block
    A1 = view(A, 1:blocksize, 1:nA)
    A2 = view(A, blocksize+1:mA, 1:nA)

    # Copy top part of A (A1) to temporary work space
    copyto!(M, A1)
    # Multiply UnitLowerTriangular V1 and A1 in-place in M (M = V1'A1)
    lmul!(V1', M)
    # Add V2'A2 to V1'A1 (M := M + V2'A2)
    mul!(M, V2', A2, one(T), one(T))
    # Multiply the elementary block loading T (UpperTriangular) and M in-place
    lmul!(H.T', M)

    # A2 := A2 - V2*M
    mul!(A2, V2, M, -one(T), one(T))
    # A1 := A1 - V1*M but since V1 is UnitLowerTriangular we do it in two steps:
    ## 1. M  := -V1*M
    ## 2. A1 := A1 + M
    lmul!(V1, M, -one(T))
    axpy!(one(T), M, A1)

    return A
end
(*)(adjH::Adjoint{T,<:HouseholderBlock{T}}, A::StridedMatrix{T}) where {T} =
        lmul!(adjH, copy(A), similar(A, (min(size(parent(adjH).V)...), size(A, 2))))

convert(::Type{Matrix}, H::Householder{T}) where {T} = lmul!(H, Matrix{T}(I, size(H, 1), size(H, 1)))
convert(::Type{Matrix{T}}, H::Householder{T}) where {T} = lmul!(H, Matrix{T}(I, size(H, 1), size(H, 1)))
