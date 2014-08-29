module EigenAuxiliary

	immutable Householder{T} <: AbstractMatrix{T}
		size::Int
		u::Vector{T}
		σ::T
	end

	function householder(A::AbstractMatrix, row::Integer, col::Integer)
		m, n = size(A)
		col < n || error("col cannot be larger than $(size(A,2)-1)")
		row >= col || error("col cannot be less than row")
		@inbounds begin
			u = A[row:end,col]
			ξ1 = u[1]
			ν = copysign(norm(u),real(ξ1))
			u[1] += ν
			ξ1ν = ξ1 + ν
			for i = 1:length(u)
				u[i] /= ξ1ν
			end
		end
		Householder(m, u, conj(ξ1ν/ν))
	end

end