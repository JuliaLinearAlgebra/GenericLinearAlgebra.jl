using LinearAlgebra
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, HermOrSym

import LinearAlgebra: lmul!, mul!

export rankUpdate!

# Rank one update

## General
### BLAS
rankUpdate!(α::T, x::StridedVector{T}, y::StridedVector{T}, A::StridedMatrix{T}) where {T<:BlasReal} = BLAS.ger!(α, x, y, A)

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
rankUpdate!(α::T, a::StridedVector{T}, A::HermOrSym{T,S}) where {T<:BlasReal,S<:StridedMatrix} = BLAS.syr!(A.uplo, α, a, A.data)
rankUpdate!(a::StridedVector{T}, A::HermOrSym{T,S}) where {T<:BlasReal,S<:StridedMatrix} = rankUpdate!(one(T), a, A)

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
rankUpdate!(α::T, A::StridedMatrix{T}, β::T, C::HermOrSym{T,S}) where {T<:BlasReal,S<:StridedMatrix} = syrk!(C.uplo, 'N', α, A, β, C.data)

## Complex
rankUpdate!(α::T, A::StridedMatrix{Complex{T}}, β::T, C::Hermitian{T,S}) where {T<:BlasReal,S<:StridedMatrix} = herk!(C.uplo, 'N', α, A, β, C.data)

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

# BLAS style mul!
## gemv
mul!(y::StridedVector{T}, A::StridedMatrix{T}, x::StridedVector{T}, α::T, β::T) where {T<:BlasFloat} = gemv!('N', α, A, x, β, y)
mul!(y::StridedVector{T}, A::Adjoint{T,<:StridedMatrix{T}}, x::StridedVector{T}, α::T, β::T) where {T<:BlasFloat} = gemv!('C', α, parent(adjA), x, β, y)

## gemm
mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::StridedMatrix{T}, α::T, β::T) where {T<:BlasFloat} = BLAS.gemm!('N', 'N', α, A, B, β, C)
mul!(C::StridedMatrix{T}, adjA::Adjoint{T,<:StridedMatrix{T}}, B::StridedMatrix{T}, α::T, β::T) where {T<:BlasFloat} = BLAS.gemm!('C', 'N', α, parent(adjA), B, β, C)
# Not optimized since it is a generic fallback. Can probably soon be removed when the signatures in base have been updated.
function mul!(C::StridedVecOrMat,
              A::StridedMatrix,
              B::StridedVecOrMat,
              α::Number,
              β::Number)

    m, n = size(C, 1), size(C, 2)
    k = size(A, 2)

    if β != 1
        if β == 0
            fill!(C, 0)
        else
            rmul!(C, β)
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
function mul!(C::StridedVecOrMat,
              adjA::Adjoint{<:Number,<:StridedMatrix},
              B::StridedVecOrMat,
              α::Number,
              β::Number)

    A = parent(adjA)
    m, n = size(C, 1), size(C, 2)
    k = size(A, 1)

    if β != 1
        if β == 0
            fill!(C, 0)
        else
            rmul!(C, β)
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

## trmm like
### BLAS versions
mul!(A::UpperTriangular{T,S}, B::StridedMatrix{T}, α::T) where {T<:BlasFloat,S} = trmm!('L', 'U', 'N', 'N', α, A.data, B)
mul!(A::LowerTriangular{T,S}, B::StridedMatrix{T}, α::T) where {T<:BlasFloat,S} = trmm!('L', 'L', 'N', 'N', α, A.data, B)
mul!(A::UnitUpperTriangular{T,S}, B::StridedMatrix{T}, α::T) where {T<:BlasFloat,S} = trmm!('L', 'U', 'N', 'U', α, A.data, B)
mul!(A::UnitLowerTriangular{T,S}, B::StridedMatrix{T}, α::T) where {T<:BlasFloat,S} = trmm!('L', 'L', 'N', 'U', α, A.data, B)
mul!(A::Adjoint{T,UpperTriangular{T,S}}, B::StridedMatrix{T}, α::T) where {T<:BlasFloat,S} = trmm!('L', 'U', 'C', 'N', α, parent(A).data, B)
mul!(A::Adjoint{T,LowerTriangular{T,S}}, B::StridedMatrix{T}, α::T) where {T<:BlasFloat,S} = trmm!('L', 'L', 'C', 'N', α, parent(A).data, B)
mul!(A::Adjoint{T,UnitUpperTriangular{T,S}}, B::StridedMatrix{T}, α::T) where {T<:BlasFloat,S} = trmm!('L', 'U', 'C', 'U', α, parent(A).data, B)
mul!(A::Adjoint{T,UnitLowerTriangular{T,S}}, B::StridedMatrix{T}, α::T) where {T<:BlasFloat,S} = trmm!('L', 'L', 'C', 'U', α, parent(A).data, B)

### Generic fallbacks
function lmul!(A::UpperTriangular{T,S}, B::StridedMatrix{T}, α::T) where {T<:Number,S}
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
function lmul!(A::LowerTriangular{T,S}, B::StridedMatrix{T}, α::T) where {T<:Number,S}
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
function lmul!(A::UnitUpperTriangular{T,S}, B::StridedMatrix{T}, α::T) where {T<:Number,S}
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
function lmul!(A::UnitLowerTriangular{T,S}, B::StridedMatrix{T}, α::T) where {T<:Number,S}
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
function lmul!(A::Adjoint{T,UpperTriangular{T,S}}, B::StridedMatrix{T}, α::T) where {T<:Number,S}
    AA = parent(A).data
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
function lmul!(A::Adjoint{T,LowerTriangular{T,S}}, B::StridedMatrix{T}, α::T) where {T<:Number,S}
    AA = parent(A).data
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
function lmul!(A::Adjoint{T,UnitUpperTriangular{T,S}}, B::StridedMatrix{T}, α::T) where {T<:Number,S}
    AA = parent(A).data
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
function lmul!(A::Adjoint{T,UnitLowerTriangular{T,S}}, B::StridedMatrix{T}, α::T) where {T<:Number,S}
    AA = parent(A).data
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