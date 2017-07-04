module JuliaBLAS

using Base.BLAS
using Base.LinAlg: BlasComplex, BlasFloat, BlasReal, HermOrSym, UnitLowerTriangular, UnitUpperTriangular

import Base.LinAlg: A_mul_B!, Ac_mul_B!

export rankUpdate!

# Rank one update

## General
### BLAS
rankUpdate!{T<:BlasReal}(α::T, x::StridedVector{T}, y::StridedVector{T}, A::StridedMatrix{T}) = ger!(α, x, y, A)

### Generic
function rankUpdate!(α::Number, x::StridedVector, y::StridedVector, A::StridedMatrix)
    m, n = size(A, 1), size(A, 2)
    m == length(x) || throw(DimensionMismatch("x vector has wrong length"))
    n == length(y) || throw(DimensionMismatch("y vector has wrong length"))
    for j = 1:n
        yjc = y[j]'
        for i = 1:m
            A[i,j] += x[i]*α*yjc
        end
    end
end

## Hermitian
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(α::T, a::StridedVector{T}, A::HermOrSym{T,S}) = syr!(A.uplo, α, a, A.data)
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(a::StridedVector{T}, A::HermOrSym{T,S}) = rankUpdate!(one(T), a, A)

### Generic
function rankUpdate!(α::Real, a::StridedVector, A::Hermitian)
    n = size(A, 1)
    n == length(a) || throw(DimensionMismatch("a vector has wrong length"))
    @inbounds for j in 1:n
        ajc = a[j]'
        for i in ((A.uplo == 'L') ? (j:n) : (1:j))
            A.data[i,j] += a[i]*α*ajc
        end
    end
    return A
end

# Rank k update
## Real
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(α::T, A::StridedMatrix{T}, β::T, C::HermOrSym{T,S}) = syrk!(C.uplo, 'N', α, A, β, C.data)

## Complex
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(α::T, A::StridedMatrix{Complex{T}}, β::T, C::Hermitian{T,S}) = herk!(C.uplo, 'N', α, A, β, C.data)

### Generic
function rankUpdate!(α::Real, A::StridedVecOrMat, C::Hermitian)
    n = size(C, 1)
    n == size(A, 1) || throw(DimensionMismatch("first dimension of A has wrong size"))
    @inbounds if C.uplo == 'L' # branch outside the loop to have larger loop to optimize
        for k in 1:size(A, 2)
            for j in 1:n
                ajkc = A[j,k]'
                for i in j:n
                    C.data[i,j] += A[i,k]*α*ajkc
                end
            end
        end
    else
        for k in 1:size(A, 2)
            for j in 1:n
                ajkc = A[j,k]'
                for i in 1:j
                    C.data[i,j] += A[i,k]*α*ajkc
                end
            end
        end
    end
    return C
end

# BLAS style A_mul_B!
## gemv
A_mul_B!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, x::StridedVector{T}, β::T, y::StridedVector{T}) = gemv!('N', α, A, x, β, y)
Ac_mul_B!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, x::StridedVector{T}, β::T, y::StridedVector{T}) = gemv!('C', α, A, x, β, y)

## gemm
A_mul_B!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, B::StridedMatrix{T}, β::T, C::StridedMatrix{T}) = gemm!('N', 'N', α, A, B, β, C)
Ac_mul_B!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, B::StridedMatrix{T}, β::T, C::StridedMatrix{T}) = gemm!('C', 'N', α, A, B, β, C)
# Not optimized since it is a generic fallback. Can probably soon be removed when the signatures in base have been updated.
function A_mul_B!(α::Number, A::StridedMatrix, B::StridedVecOrMat, β::Number, C::StridedVecOrMat)
    m, n = size(C, 1), size(C, 2)
    k = size(A, 2)

    if β != 1
        if β == 0
            fill!(C, 0)
        else
            scale!(C, β)
        end
    end
    for j = 1:n
        for i = 1:m
            for l = 1:k
                C[i,j] += α*A[i,l]*B[l,j]
            end
        end
    end
    return C
end
function Ac_mul_B!(α::Number, A::StridedMatrix, B::StridedVecOrMat, β::Number, C::StridedVecOrMat)
    m, n = size(C, 1), size(C, 2)
    k = size(A, 1)

    if β != 1
        if β == 0
            fill!(C, 0)
        else
            scale!(C, β)
        end
    end
    for j = 1:n
        for i = 1:m
            for l = 1:k
                C[i,j] += α*A[l,i]'*B[l,j]
            end
        end
    end
    return C
end

## trmm
### BLAS versions
A_mul_B!{T<:BlasFloat,S}(α::T, A::UpperTriangular{T,S}, B::StridedMatrix{T}) = trmm!('L', 'U', 'N', 'N', α, A.data, B)
A_mul_B!{T<:BlasFloat,S}(α::T, A::LowerTriangular{T,S}, B::StridedMatrix{T}) = trmm!('L', 'L', 'N', 'N', α, A.data, B)
A_mul_B!{T<:BlasFloat,S}(α::T, A::UnitUpperTriangular{T,S}, B::StridedMatrix{T}) = trmm!('L', 'U', 'N', 'U', α, A.data, B)
A_mul_B!{T<:BlasFloat,S}(α::T, A::UnitLowerTriangular{T,S}, B::StridedMatrix{T}) = trmm!('L', 'L', 'N', 'U', α, A.data, B)
Ac_mul_B!{T<:BlasFloat,S}(α::T, A::UpperTriangular{T,S}, B::StridedMatrix{T}) = trmm!('L', 'U', 'C', 'N', α, A.data, B)
Ac_mul_B!{T<:BlasFloat,S}(α::T, A::LowerTriangular{T,S}, B::StridedMatrix{T}) = trmm!('L', 'L', 'C', 'N', α, A.data, B)
Ac_mul_B!{T<:BlasFloat,S}(α::T, A::UnitUpperTriangular{T,S}, B::StridedMatrix{T}) = trmm!('L', 'U', 'C', 'U', α, A.data, B)
Ac_mul_B!{T<:BlasFloat,S}(α::T, A::UnitLowerTriangular{T,S}, B::StridedMatrix{T}) = trmm!('L', 'L', 'C', 'U', α, A.data, B)

### Generic fallbacks
function A_mul_B!{T<:Number,S}(α::T, A::UpperTriangular{T,S}, B::StridedMatrix{T})
    AA = A.data
    m, n = size(B)
    for i = 1:m
        for j = 1:n
            B[i,j] = α*AA[i,i]*B[i,j]
            for l = i + 1:m
                B[i,j] += α*AA[i,l]*B[l,j]
            end
        end
    end
    return B
end
function A_mul_B!{T<:Number,S}(α::T, A::LowerTriangular{T,S}, B::StridedMatrix{T})
    AA = A.data
    m, n = size(B)
    for i = m:-1:1
        for j = 1:n
            B[i,j] = α*AA[i,i]*B[i,j]
            for l = 1:i - 1
                B[i,j] += α*AA[i,l]*B[l,j]
            end
        end
    end
    return B
end
function A_mul_B!{T<:Number,S}(α::T, A::UnitUpperTriangular{T,S}, B::StridedMatrix{T})
    AA = A.data
    m, n = size(B)
    for i = 1:m
        for j = 1:n
            B[i,j] = α*B[i,j]
            for l = i + 1:m
                B[i,j] = α*AA[i,l]*B[l,j]
            end
        end
    end
    return B
end
function A_mul_B!{T<:Number,S}(α::T, A::UnitLowerTriangular{T,S}, B::StridedMatrix{T})
    AA = A.data
    m, n = size(B)
    for i = m:-1:1
        for j = 1:n
            B[i,j] = α*B[i,j]
            for l = 1:i - 1
                B[i,j] += α*AA[i,l]*B[l,j]
            end
        end
    end
    return B
end
function Ac_mul_B!{T<:Number,S}(α::T, A::UpperTriangular{T,S}, B::StridedMatrix{T})
    AA = A.data
    m, n = size(B)
    for i = m:-1:1
        for j = 1:n
            B[i,j] = α*AA[i,i]*B[i,j]
            for l = 1:i - 1
                B[i,j] += α*AA[l,i]'*B[l,j]
            end
        end
    end
    return B
end
function Ac_mul_B!{T<:Number,S}(α::T, A::LowerTriangular{T,S}, B::StridedMatrix{T})
    AA = A.data
    m, n = size(B)
    for i = 1:m
        for j = 1:n
            B[i,j] = α*AA[i,i]*B[i,j]
            for l = i + 1:m
                B[i,j] += α*AA[l,i]'*B[l,j]
            end
        end
    end
    return B
end
function Ac_mul_B!{T<:Number,S}(α::T, A::UnitUpperTriangular{T,S}, B::StridedMatrix{T})
    AA = A.data
    m, n = size(B)
    for i = m:-1:1
        for j = 1:n
            B[i,j] = α*B[i,j]
            for l = 1:i - 1
                B[i,j] += α*AA[l,i]'*B[l,j]
            end
        end
    end
    return B
end
function Ac_mul_B!{T<:Number,S}(α::T, A::UnitLowerTriangular{T,S}, B::StridedMatrix{T})
    AA = A.data
    m, n = size(B)
    for i = 1:m
        for j = 1:n
            B[i,j] = α*B[i,j]
            for l = i + 1:m
                B[i,j] = α*AA[l,i]'*B[l,j]
            end
        end
    end
    return B
end

end