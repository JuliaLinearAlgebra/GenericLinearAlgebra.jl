module JuliaBLAS

using Base.BLAS
using Base.LinAlg: BlasReal

export rankUpdate!

# Rank one update
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(α::T, a::StridedVector{T}, A::Symmetric{T,S}) = syr!(A.uplo, α, a, A.data)
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(a::StridedVector{T}, A::Symmetric{T,S}) = rankUpdate!(one(T), a, A)

# Rank k update
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(α::T, A::StridedMatrix{T}, β::T, C::Symmetric{T,S}) = syrk!(C.uplo, 'N', α, A, β, C.data)
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(α::T, A::StridedMatrix{T}, C::Symmetric{T,S}) = rankUpdate!(α, A, 
	one(T), C)
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(A::StridedMatrix{T}, C::Symmetric{T,S}) = rankUpdate!(one(T), A, one(T), C)

end