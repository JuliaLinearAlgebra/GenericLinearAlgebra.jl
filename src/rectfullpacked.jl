# Rectangular Full Packed Matrices

import Base: \
import Base.LinAlg: BlasFloat
import ..GenericLinearAlgebra.LAPACK2: trttf!

struct HermitianRFP{T<:BlasFloat} <: AbstractMatrix{T}
    data::Vector{T}
    transr::Char
    uplo::Char
end

function Base.size(A::HermitianRFP, i::Integer)
    if i == 1 || i == 2
        return (isqrt(8*length(A.data) + 1) - 1) >> 1
    elseif i > 2
        return 1
    else
        return size(A.data, i)
    end
end
function Base.size(A::HermitianRFP)
    n = size(A, 1)
    return (n, n)
end

function Base.getindex(A::HermitianRFP, i::Integer, j::Integer)
    n = size(A, 1)
    n2 = n >> 1
    if i < 1 || i > n || j < 0 || j > n
        throw(BoundsError(A, (i,j)))
    end
    if A.uplo == 'L'
        if i < j
            return conj(A[j,i])
        end

        if j <= n2 + isodd(n)
            if A.transr == 'N'
                return A.data[i + iseven(n) + (j - 1)*(n + iseven(n))]
            else
                return conj(A.data[(i - 1)*(n + iseven(n)) + j + iseven(n)])
            end
        else
            if A.transr == 'N'
                return conj(A.data[(i - n2 - 1)*(n + iseven(n)) + j - n2 - isodd(n)])
            else
                return A.data[i - n2 - isodd(n) + (j - n2 - 1)*(n + iseven(n))]
            end
        end
    else
        if i > j
            return conj(A[j,i])
        end

        if j > n2
            if A.transr == 'N'
                return A.data[i + (j - n2 - 1)*(n + iseven(n))]
            else
                return conj(A.data[(i - n2 - 1)*(n + iseven(n)) + j])
            end
        else
            if A.transr == 'N'
                return conj(A.data[(i - 1)*(n + iseven(n)) + j + n2 + 1])
            else
                return A.data[i - n2 - isodd(n) + (j - n2 - 1)*(n + iseven(n))]
            end
        end
    end
end

function Ac_mul_A_RFP(A::Matrix{T}, uplo = :U) where T<:BlasFloat
    n = size(A, 2)
    if uplo == :U
        C = LAPACK2.sfrk!('N', 'U', T <: Complex ? 'C' : 'T', 1.0, A, 0.0, Vector{T}(n*(n + 1) >> 1))
        return HermitianRFP(C, 'N', 'U')
    elseif uplo == :L
        C = LAPACK2.sfrk!('N', 'L', T <: Complex ? 'C' : 'T', 1.0, A, 0.0, Vector{T}(n*(n + 1) >> 1))
       return  HermitianRFP(C, 'N', 'L')
   else
        throw(ArgumentError("uplo must be either :L or :U"))
    end
end

Base.copy(A::HermitianRFP) = HermitianRFP(copy(A.data), A.transr, A.uplo)

struct TriangularRFP{T<:BlasFloat} <: AbstractMatrix{T}
    data::Vector{T}
    transr::Char
    uplo::Char
end

function TriangularRFP(A::StridedMatrix, uplo::Symbol = :U)
    if uplo == :U
        return TriangularRFP(trttf!('N', 'U', A), 'N', 'U')
    elseif uplo == :L
        return TriangularRFP(trttf!('N', 'L', A), 'N', 'L')
    else
        throw(ArgumentError("uplo must be either :U or :L but was :$uplo"))
    end
end

function Base.size(A::TriangularRFP, i::Integer)
    if i == 1 || i == 2
        return (isqrt(8*length(A.data) + 1) - 1) >> 1
    elseif i > 2
        return 1
    else
        return size(A.data, i)
    end
end
function Base.size(A::TriangularRFP)
    n = size(A, 1)
    return (n, n)
end

Base.copy(A::TriangularRFP) = TriangularRFP(copy(A.data), A.transr, A.uplo)

function Base.full(A::TriangularRFP)
    C = LAPACK2.tfttr!(A.transr, A.uplo, A.data)
    if A.uplo == 'U'
        return triu!(C)
    else
        return tril!(C)
    end
end

Base.LinAlg.inv!(A::TriangularRFP) = TriangularRFP(LAPACK2.tftri!(A.transr, A.uplo, 'N', A.data), A.transr, A.uplo)
Base.LinAlg.inv(A::TriangularRFP)  = Base.LinAlg.inv!(copy(A))

A_ldiv_B!(A::TriangularRFP{T}, B::StridedVecOrMat{T}) where T =
    LAPACK2.tfsm!(A.transr, 'L', A.uplo, 'N', 'N', one(T), A.data, B)
(\)(A::TriangularRFP, B::StridedVecOrMat) = A_ldiv_B!(A, copy(B))

struct CholeskyRFP{T<:BlasFloat} <: Factorization{T}
    data::Vector{T}
    transr::Char
    uplo::Char
end

Base.LinAlg.cholfact!(A::HermitianRFP{T}) where {T<:BlasFloat} = CholeskyRFP(LAPACK2.pftrf!(A.transr, A.uplo, copy(A.data)), A.transr, A.uplo)
Base.LinAlg.cholfact(A::HermitianRFP{T}) where {T<:BlasFloat} = cholfact!(copy(A))
Base.LinAlg.factorize(A::HermitianRFP) = cholfact(A)

Base.copy(F::CholeskyRFP{T}) where T = CholeskyRFP{T}(copy(F.data), F.transr, F.uplo)

# Solve
(\)(A::CholeskyRFP, B::StridedVecOrMat) = LAPACK2.pftrs!(A.transr, A.uplo, A.data, copy(B))
(\)(A::HermitianRFP, B::StridedVecOrMat) = cholfact(A)\B

Base.LinAlg.inv!(A::CholeskyRFP) = HermitianRFP(LAPACK2.pftri!(A.transr, A.uplo, A.data), A.transr, A.uplo)
Base.LinAlg.inv(A::CholeskyRFP)  = Base.LinAlg.inv!(copy(A))
Base.LinAlg.inv(A::HermitianRFP) = Base.LinAlg.inv!(cholfact(A))
