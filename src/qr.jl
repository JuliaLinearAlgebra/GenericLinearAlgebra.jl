module QRModule

	using ..Sym
	using ..HouseholderModule: Householder, HouseholderBlock, householder!
	using Base.LinAlg: QR, axpy!
	using ArrayViews

	import Base: getindex, size

	immutable QR2{T,S<:AbstractMatrix{T},U} <: Factorization{T}
		data::S
		reflectors::Vector{U}
	end

	immutable Q{T,S<:QR2{T}} <: AbstractMatrix{T}
		data::S
	end

	function qrUnblocked!{T}(A::DenseMatrix{T})
		m, n = size(A)
		minmn = min(m,n)
		τ = Array(Householder{T}, minmn)
		for i = 1:min(m,n)
			H = householder!(view(A,i,i), view(A, i+1:m, i))
			τ[i] = H
			Ac_mul_B!(H, view(A, i:m, i + 1:n))
		end
		QR2{T,typeof(A),eltype(τ)}(A, τ)
	end

	getindex{T,S,U<:Householder}(A::QR2{T,S,U}, ::Type{Sym{:Q}}) = Q{T,typeof(A)}(A)
	function getindex{T,S,U<:Householder}(A::QR2{T,S,U}, ::Type{Sym{:R}}) 
		m, n = size(A)
		m >= n ? Triangular(view(A.data, 1:n, 1:n), :U) : error("R matrix is trapezoid and cannot be extracted with indexing")
	end

	# This method extracts the Q part of the factorization in the block form H = I - VTV' where V is a matrix the reflectors and T is an upper triangular matrix
	function getindex{T,S,U}(A::QR2{T,S,U}, ::Type{Sym{:QBlocked}})
		m, n = size(A)
		D = A.data
		ref = A.reflectors
		Tmat = Array(T, n, n)
		Tmat[1,1] = ref[1].τ
		for i = 2:n
			Tmat[i,i] = ref[i].τ
			t12 = view(Tmat, 1:i-1, i)
			Ac_mul_B!(-one(T), view(D, i+1:m, 1:i-1), view(D, i+1:m, i), zero(T), t12)
			axpy!(-one(T), view(D, i, 1:i-1), t12)
			A_mul_B!(Triangular(view(Tmat, 1:i-1, 1:i-1), :U), t12)
			scale!(t12, Tmat[i,i])
		end
		HouseholderBlock{T,typeof(D)}(D, Triangular(Tmat, :U))
	end

	size(A::QR2) = size(A.data)
	size(A::QR2, i::Integer) = size(A.data, i)

	function qrBlocked!(A::DenseMatrix, blocksize::Integer, work = Array(eltype(A), blocksize, size(A, 2)))
		m, n = size(A)
		A1 = view(A, 1:m, 1:min(n, blocksize))
		F = qrUnblocked!(A1)
		if n > blocksize
			A2 = view(A, 1:m, blocksize + 1:n)
			Ac_mul_B!(F[Sym{:QBlocked}], A2, view(work, 1:blocksize, 1:n - blocksize))
			qrBlocked!(view(A, blocksize + 1:m, blocksize + 1:n), blocksize, work)
		end
		A
	end

	function qrTiledUnblocked!{T,S<:DenseMatrix}(A::Triangular{T,S,:U,false}, B::DenseMatrix)
		m, n = size(B)
		Ad = A.data
		for i = 1:m
			H = householder!(view(Ad,i,i), view(B,1:m,i))
			Ac_mul_B!(H, )
		end
	end
end