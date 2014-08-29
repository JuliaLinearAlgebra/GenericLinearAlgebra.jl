module CholeskyModule

	using ..JuliaBLAS
	using Base.LinAlg: A_rdiv_Bc!, chksquare
	using ArrayViews

	function cholUnblocked!{T<:Number}(A::AbstractMatrix{T}, ::Type{:L})
		n = chksquare(A)
		A[1,1] = sqrt(A[1,1])
		if n > 1
			a21 = view(A, 2:n, 1)
			scale!(a21, inv(A[1,1]))

			A22 = view(A, 2:n, 2:n)
			rankUpdate!(-one(T), a21, Symmetric(A22, :L))
			cholUnblocked!(A22, :L)
		end
		A
	end

	function cholBlocked!{T<:Number}(A::AbstractMatrix{T}, ::Type{:L}, blocksize::Integer)
		n = chksquare(A)
		mnb = min(n, blocksize)
		A11 = view(A, 1:mnb, 1:mnb)
		cholUnblocked!(A11, :L)
		if n > blocksize
			A21 = view(A, blocksize+1:n, 1:blocksize)
			A_rdiv_Bc!(A21, Triangular(A11, :L))

			A22 = view(A, blocksize+1:n, blocksize+1:n)
			rankUpdate!(-one(T), A21, Symmetric(A22, :L))
			cholBlocked!(A22, :L, blocksize)
		end
		A
	end
end