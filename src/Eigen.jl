module Eigen
import Base: size
import Base.LinAlg: chksquare, eigvals, eigvals!, elementaryLeft!, givensAlgorithm

immutable SymmetricTridiagonalFactorization{T} <: Factorization{T}
	factors::Matrix{T}
	scalarfactors::Vector{T}
	diagonals::SymTridiagonal
end

symtri!(A::Hermitian) = A.uplo == 'L' ? symtriLower!(A.S) : symtriUpper!(A.S)

function symtriLower!{T}(AS::Matrix{T}) # Assume that lower triangle stores the relevant part
	n = size(AS,1)
	τ = zeros(T,n-1)
	u = Array(T,n,1)
	@inbounds begin
		for k = 1:n-2+!(T<:Real)
			τk = elementaryLeft!(AS,k+1,k)
			τ[k] = τk
	
			for i = k+1:n u[i] = AS[i,k+1] end
			for j = k+2:n
				ASjk = AS[j,k]
				for i = j:n
					u[i] += AS[i,j]*ASjk
				end
			end
			for j = k+1:n-1
				tmp = zero(T)
				for i = j+1:n
					tmp += AS[i,j]'AS[i,k]
				end
				u[j] += tmp
			end
	
			vcAv = u[k+1]
			for i = k+2:n vcAv += AS[i,k]'u[i] end
			ξτ2 = real(vcAv)*abs2(τk)/2
	
			u[k+1] = u[k+1]*τk - ξτ2
			for i = k+2:n 
				u[i] = u[i]*τk - ξτ2*AS[i,k]
			end
	
			AS[k+1,k+1] -= 2real(u[k+1])
			for i = k+2:n AS[i,k+1] -= u[i] + AS[i,k]*u[k+1]' end
			for j = k+2:n
				ASjk = AS[j,k]
				uj = u[j]
				AS[j,j] -= 2real(uj*ASjk')
				for i = j+1:n
					AS[i,j] -= u[i]*ASjk' + AS[i,k]*uj'
				end
			end
		end
	end
	SymmetricTridiagonalFactorization(AS,τ,SymTridiagonal(real(diag(AS)),real(diag(AS,-1))))
end

eigvals(A::SymmetricTridiagonalFactorization, l::Real=1, h::Real=size(A,1)) = eigvals(A.diagonals, l, h)
eigvals!(A::Hermitian, l::Real=1, h::Real=size(A,1)) = eigvals(symtri!(A), l, h)

function hessfact!{T}(A::AbstractMatrix{T})
	n = chksquare(A)
	τ = zeros(T, n-1)
	for k = 1:n-2+!(T<:Real)
		τk = elementaryLeft!(A,k+1,k)
		τ[k] = τk

		for j = k+1:n
			vAj = A[k+1,j]
			for i = k+2:n
				vAj += conj(A[i,k])*A[i,j]
			end
			vAj = conj(τk)*vAj
			A[k+1,j] -= vAj
			for i = k+2:n
				A[i,j] -= A[i,k]*vAj
			end
		end

		for i = 1:n
			Avi = A[i,k+1]
			for j = k+2:n
				Avi += A[i,j]*A[j,k]
			end
			Avi = Avi*τk
			A[i,k+1] -= Avi
			for j = k+2:n
				A[i,j] -= Avi*conj(A[j,k])
			end
		end
	end
	return Hessenberg(A,τ)
end

# Non-commuting givens
# function givensAlgorithm(f::Number,g::Number)
# 	c = abs2(f)
# 	s = f'g
# 	as = abs(s)
# 	csmax = max(c,as)
# 	csmin = min(c,as)
# 	nminv = inv(csmax*sqrt(1+csmin/csmax))
# 	c *= nminv
# 	s *= nminv
# 	return c, s, c*f+s*g
# end

size(H::Hessenberg, args...) = size(H.factors, args...)

function schurSingleShift!{T}(H::Hessenberg{T}; tol = sqrt(eps(real(one(T)))))
	n = size(H, 1)
	for m = n:-1:2
		while true
			singleShiftQR!(H, H.factors[m,m], m)
			if abs(H.factors[m,m-1]) < tol*abs(H.factors[m,m]) break end
		end
	end
	return H.factors
end

function singleShiftQR!(H::Hessenberg, shift::Number, m::Integer = size(H, 1))
	G = givens(H.factors[1,1] - shift,H.factors[2,1],1,2,size(H, 1))
	A_mul_B!(G,H.factors)
	A_mul_Bc!(H.factors,G)
	# A_mul_B!(G,H.R)
	for i = 1:m-2
		G = givens(H.factors,i+1,i+2,i)
		A_mul_B!(G,H.factors)
		A_mul_Bc!(H.factors,G)
		# A_mul_B!(G,H.R)
	end
	return H
end

end