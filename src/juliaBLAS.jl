using LinearAlgebra
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, HermOrSym

import LinearAlgebra: lmul!, mul!

export rankUpdate!

# Rank one update

## General
### BLAS
rankUpdate!(
    A::StridedMatrix{T},
    x::StridedVector{T},
    y::StridedVector{T},
    α::T,
) where {T<:BlasReal} = BLAS.ger!(α, x, y, A)

### Generic
function rankUpdate!(A::StridedMatrix, x::StridedVector, y::StridedVector, α::Number)
    m, n = size(A, 1), size(A, 2)
    m == length(x) || throw(DimensionMismatch("x vector has wrong length"))
    n == length(y) || throw(DimensionMismatch("y vector has wrong length"))
    for j = 1:n
        yjc = y[j]'
        for i = 1:m
            A[i, j] += x[i] * α * yjc
        end
    end
end

## Hermitian
function rankUpdate!(
    A::HermOrSym{T,S},
    a::StridedVector{T},
    α::T,
) where {T<:BlasReal,S<:StridedMatrix}
    BLAS.syr!(A.uplo, α, a, A.data)
    return A
end
function rankUpdate!(
    A::Hermitian{Complex{T},S},
    a::StridedVector{Complex{T}},
    α::T,
) where {T<:BlasReal,S<:StridedMatrix}
    BLAS.her!(A.uplo, α, a, A.data)
    return A
end
rankUpdate!(A::HermOrSym{T,S}, a::StridedVector{T}) where {T<:BlasFloat,S<:StridedMatrix} =
    rankUpdate!(A, a, one(real(T)))

### Generic
function rankUpdate!(A::Hermitian, a::StridedVector, α::Real)
    n = size(A, 1)
    n == length(a) || throw(DimensionMismatch("a vector has wrong length"))
    @inbounds for j = 1:n
        ajc = a[j]'
        for i in ((A.uplo == 'L') ? (j:n) : (1:j))
            A.data[i, j] += a[i] * α * ajc
        end
    end
    return A
end

# Rank k update
## Real
function rankUpdate!(
    C::HermOrSym{T,S},
    A::StridedMatrix{T},
    α::T,
    β::T,
) where {T<:BlasReal,S<:StridedMatrix}
    BLAS.syrk!(C.uplo, 'N', α, A, β, C.data)
    return C
end

## Complex
function rankUpdate!(
    C::Hermitian{Complex{T},S},
    A::StridedMatrix{Complex{T}},
    α::T,
    β::T,
) where {T<:BlasReal,S<:StridedMatrix}
    BLAS.herk!(C.uplo, 'N', α, A, β, C.data)
    return C
end

### Generic
function rankUpdate!(C::Hermitian, A::StridedVecOrMat, α::Real)
    n = size(C, 1)
    n == size(A, 1) || throw(DimensionMismatch("first dimension of A has wrong size"))
    @inbounds if C.uplo == 'L' # branch outside the loop to have larger loop to optimize
        for k = 1:size(A, 2)
            for j = 1:n
                ajkc = A[j, k]'
                for i = j:n
                    C.data[i, j] += A[i, k] * α * ajkc
                end
            end
        end
    else
        for k = 1:size(A, 2)
            for j = 1:n
                ajkc = A[j, k]'
                for i = 1:j
                    C.data[i, j] += A[i, k] * α * ajkc
                end
            end
        end
    end
    return C
end

### Generic fallbacks
function lmul!(A::UpperTriangular{T,S}, B::StridedVecOrMat{T}, α::T) where {T<:Number,S}
    AA = A.data
    m, n = size(B, 1), size(B, 2)
    for i ∈ 1:m
        for j ∈ 1:n
            B[i, j] = α * AA[i, i] * B[i, j]
            for l ∈ (i+1):m
                B[i, j] += α * AA[i, l] * B[l, j]
            end
        end
    end
    return B
end
function lmul!(A::LowerTriangular{T,S}, B::StridedVecOrMat{T}, α::T) where {T<:Number,S}
    AA = A.data
    m, n = size(B, 1), size(B, 2)
    for i ∈ m:-1:1
        for j ∈ 1:n
            B[i, j] = α * AA[i, i] * B[i, j]
            for l ∈ 1:(i-1)
                B[i, j] += α * AA[i, l] * B[l, j]
            end
        end
    end
    return B
end
function lmul!(A::UnitUpperTriangular{T,S}, B::StridedVecOrMat{T}, α::T) where {T<:Number,S}
    AA = A.data
    m, n = size(B, 1), size(B, 2)
    for i ∈ 1:m
        for j ∈ 1:n
            B[i, j] = α * B[i, j]
            for l ∈ (i+1):m
                B[i, j] += α * AA[i, l] * B[l, j]
            end
        end
    end
    return B
end
function lmul!(A::UnitLowerTriangular{T,S}, B::StridedVecOrMat{T}, α::T) where {T<:Number,S}
    AA = A.data
    m, n = size(B, 1), size(B, 2)
    for i ∈ m:-1:1
        for j ∈ 1:n
            B[i, j] = α * B[i, j]
            for l ∈ 1:(i-1)
                B[i, j] += α * AA[i, l] * B[l, j]
            end
        end
    end
    return B
end
function lmul!(
    A::LowerTriangular{T,Adjoint{T,S}},
    B::StridedMatrix{T},
    α::T,
) where {T<:Number,S}
    AA = parent(A.data)
    m, n = size(B)
    for i ∈ m:-1:1
        for j ∈ 1:n
            B[i, j] = α * AA[i, i]' * B[i, j]
            for l ∈ 1:(i-1)
                B[i, j] += α * AA[l, i]' * B[l, j]
            end
        end
    end
    return B
end
function lmul!(
    A::UpperTriangular{T,Adjoint{T,S}},
    B::StridedMatrix{T},
    α::T,
) where {T<:Number,S}
    AA = parent(A.data)
    m, n = size(B)
    for i ∈ 1:m
        for j ∈ 1:n
            B[i, j] = α * AA[i, i]' * B[i, j]
            for l ∈ (i+1):m
                B[i, j] += α * AA[l, i]' * B[l, j]
            end
        end
    end
    return B
end
function lmul!(
    A::UnitLowerTriangular{T,Adjoint{T,S}},
    B::StridedMatrix{T},
    α::T,
) where {T<:Number,S}
    AA = parent(A.data)
    m, n = size(B)
    for i ∈ m:-1:1
        for j ∈ 1:n
            B[i, j] = α * B[i, j]
            for l ∈ 1:(i-1)
                B[i, j] += α * AA[l, i]' * B[l, j]
            end
        end
    end
    return B
end
function lmul!(
    A::UnitUpperTriangular{T,Adjoint{T,S}},
    B::StridedMatrix{T},
    α::T,
) where {T<:Number,S}
    AA = parent(A.data)
    m, n = size(B)
    for i ∈ 1:m
        for j ∈ 1:n
            B[i, j] = α * B[i, j]
            for l ∈ (i+1):m
                B[i, j] += α * AA[l, i]' * B[l, j]
            end
        end
    end
    return B
end
