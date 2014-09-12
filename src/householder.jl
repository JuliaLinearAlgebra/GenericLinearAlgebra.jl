module HouseholderModule

	using ..JuliaBLAS: rankUpdate!
	using Base.LinAlg: BlasReal, axpy!
	using ArrayViews

	import Base: convert, full, size
	import Base.LinAlg: A_mul_B!, Ac_mul_B!

	immutable Householder{T,S<:DenseVector}
		v::S
		τ::T
	end
	immutable HouseholderBlock{T,S<:DenseMatrix}
		V::S
		T::Triangular{T}
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

	# See zlarfg.f
	function householder!{T,S}(α::ContiguousView{T,0,S}, x::DenseVector{T})
    	m = length(x)
    	xnorm = norm(x)
    	α1 = α[1]
    	β = -copysign(lapy(α[1], xnorm), real(α1))
    	safmin = realmin(β)/eps(typeof(β))
    	knt = 0
    	if abs(β) < safmin
    		
    		# xnorm, β may be inaccurate; scale x and recompute them
    		rsafmin = inv(safmin)
    		while true
    			knt += 1
    			scale!(x, rsafmin)
    			β *= rsafmin
    			α1 *= rsafmin
    			abs(β) < safmin || break
    		end
    		
    		# new β is at most 1, at least safmin
    		xnorm = norm(x)
    		β = -sign(lapy(α1, xnorm), real(α1))
    	end
    	τ = (β - α1)/β
    	scale!(x, inv(α1 - β))
    	
    	# if α is subnormal, it may lose relative accuracy
    	for i = 1:knt
    		β *= safmin
    	end
    	α[1] = β

    	Householder{eltype(x),typeof(x)}(x, τ)
	end

	size(H::Householder) = (length(H.v), length(H.v))
	size(H::Householder, i::Integer) = i <= 2 ? length(H.v) : 1

	function A_mul_B!(H::Householder, A::DenseMatrix)
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

	function A_mul_B!(A::DenseMatrix, H::Householder)
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

	function Ac_mul_B!(H::Householder, A::DenseMatrix)
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

	function A_mul_B!{T}(H::HouseholderBlock{T}, A::DenseMatrix{T}, M::DenseMatrix{T})
		V = H.V
		mA, nA = size(A)
		nH, blocksize = size(V)
		nH == mA || throw(DimensionMismatch(""))

		V1 = Triangular(view(V, 1:blocksize, 1:blocksize), :L, true)
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

	function Ac_mul_B!{T}(H::HouseholderBlock{T}, A::DenseMatrix{T}, M::DenseMatrix)
		V = H.V
		mA, nA = size(A)
		nH, blocksize = size(V)
		nH == mA || throw(DimensionMismatch(""))

		V1 = Triangular(view(V, 1:blocksize, 1:blocksize), :L, true)
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

	convert{T}(::Type{Matrix}, H::Householder{T}) = A_mul_B!(H, eye(T, size(H, 1)))
	convert{T}(::Type{Matrix{T}}, H::Householder{T}) = A_mul_B!(H, eye(T, size(H, 1)))

	full(H::Householder) = convert(Matrix, H)
end