module JuliaBLAS

using Base.BLAS
using Base.LinAlg: BlasReal, BlasFloat

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
            A[i,j] += α*x[i]*yjc
        end
    end
end

## Symmetric
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(α::T, a::StridedVector{T}, A::Symmetric{T,S}) = syr!(A.uplo, α, a, A.data)
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(a::StridedVector{T}, A::Symmetric{T,S}) = rankUpdate!(one(T), a, A)

# Rank k update
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(α::T, A::StridedMatrix{T}, β::T, C::Symmetric{T,S}) = syrk!(C.uplo, 'N', α, A, β, C.data)
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(α::T, A::StridedMatrix{T}, C::Symmetric{T,S}) = rankUpdate!(α, A, 
	one(T), C)
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(A::StridedMatrix{T}, C::Symmetric{T,S}) = rankUpdate!(one(T), A, one(T), C)

# BLAS style A_mul_B!
## gemv
A_mul_B!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, x::StridedVector{T}, β::T, y::StridedVector{T}) = gemv!('N', α, A, x, β, y)
Ac_mul_B!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, x::StridedVector{T}, β::T, y::StridedVector{T}) = gemv!('C', α, A, x, β, y)

## gemm
A_mul_B!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, B::StridedMatrix{T}, β::T, C::StridedMatrix{T}) = gemm!('N', 'N', α, A, B, β, C)
Ac_mul_B!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, B::StridedMatrix{T}, β::T, C::StridedMatrix{T}) = gemm!('C', 'N', α, A, B, β, C)

## trmm
A_mul_B!{T<:BlasFloat,S}(α::T, A::Triangular{T,S,:U,false}, B::StridedMatrix{T}) = trmm!('L', 'U', 'N', 'N', α, A.data, B)
A_mul_B!{T<:BlasFloat,S}(α::T, A::Triangular{T,S,:L,false}, B::StridedMatrix{T}) = trmm!('L', 'L', 'N', 'N', α, A.data, B)
A_mul_B!{T<:BlasFloat,S}(α::T, A::Triangular{T,S,:U,true}, B::StridedMatrix{T}) = trmm!('L', 'U', 'N', 'U', α, A.data, B)
A_mul_B!{T<:BlasFloat,S}(α::T, A::Triangular{T,S,:L,true}, B::StridedMatrix{T}) = trmm!('L', 'L', 'N', 'U', α, A.data, B)
Ac_mul_B!{T<:BlasFloat,S}(α::T, A::Triangular{T,S,:U,false}, B::StridedMatrix{T}) = trmm!('L', 'U', 'C', 'N', α, A.data, B)
Ac_mul_B!{T<:BlasFloat,S}(α::T, A::Triangular{T,S,:L,false}, B::StridedMatrix{T}) = trmm!('L', 'L', 'C', 'N', α, A.data, B)
Ac_mul_B!{T<:BlasFloat,S}(α::T, A::Triangular{T,S,:U,true}, B::StridedMatrix{T}) = trmm!('L', 'U', 'C', 'U', α, A.data, B)
Ac_mul_B!{T<:BlasFloat,S}(α::T, A::Triangular{T,S,:L,true}, B::StridedMatrix{T}) = trmm!('L', 'L', 'C', 'U', α, A.data, B)
end