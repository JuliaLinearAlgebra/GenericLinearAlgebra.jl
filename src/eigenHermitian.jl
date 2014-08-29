module EigenHermitian

	using Base.LinAlg: givensAlgorithm

	# type SymmetricTridiagonalFactorization{T} <: Factorization{T}
	# 	S::SymTridiagonal{T}
	# 	# bulge::T
	# 	# bulgeindex::Int
	# 	R::Base.LinAlg.Rotation
	# end

	immutable SymmetricTridiagonalFactorization{T} <: Factorization{T}
		factors::Matrix{T}
		scalarfactors::Vector{T}
		diagonals::SymTridiagonal
	end

	# function symmetricTridiagonal!{T}(A::AbstractMatrix{T})
	# 	n = chksquare(A)
	# 	R = Base.LinAlg.Rotation(AbstractMatrix[])
	# 	for i = 1:n-2
	# 		H = householder(A,i+1,i)
	# 		A_mul_B!(H,A)
	# 		A_mul_Bc!(A,H)
	# 		A_mul_B!(H,R)
	# 	end
	# 	return SymmetricTridiagonalFactorization(SymTridiagonal(real(diag(A)),real(diag(A,1))), R)
	# end

	function eigvals2x2(d1,d2,e)
		r1 = 0.5*(d1+d2)
		r2 = 0.5hypot(d1-d2,2*e)
		return r1 + r2, r1 - r2
	end

	function eigvalsPWK!{T<:FloatingPoint}(S::SymTridiagonal{T}; tol = abs2(eps(T)), debug::Bool=false)
		d = S.dv
		e = S.ev
		n = length(d)
		blockstart = 1
		blockend = n
		@inbounds begin
			for i = 1:n-1 e[i] = abs2(e[i]) end
			while true
				for blockend = blockstart+1:n
					# if abs(e[blockend-1]) <= tol*(abs(d[blockend-1]) + abs(d[blockend]))
					if abs(e[blockend-1]) == 0
						blockend -= 1
						break 
					end
				end
				if blockstart == blockend
					blockstart += 1
				elseif blockstart + 1 == blockend
					d[blockstart], d[blockend] = eigvals2x2(d[blockstart],d[blockend],sqrt(e[blockstart]))
					e[blockstart] = zero(T)
					blockstart += 1
				else
					# if abs(d[blockstart]) > abs(d[blockend])
						sqrte = sqrt(e[blockstart])
						μ = (d[blockstart+1]-d[blockstart])/(2*sqrte)
						r = hypot(μ,one(T))
						μ = d[blockstart] - sqrte/(μ+copysign(r,μ))
						singleShiftQLPWK!(S,μ,blockstart,blockend)
						debug && @printf("QL, blockstart: %d, blockend: %d, e[blockstart]: %e, e[blockend-1]:%e, μ: %f\n", blockstart, blockend, e[blockstart], e[blockend-1], μ)
					# else
						# δ = 0.5*(d[blockend-1]-d[blockend])
						# μ = d[blockend] - copysign(abs2(e[blockend-1]),δ)/(abs(δ) + hypot(δeblockend-1]))
						# μ = (d[blockend-1]-d[blockend])/(2*e[blockend-1])
						# r = hypot(μ,one(T))
						# μ = d[blockend] - (e[blockend-1]/(μ+copysign(r,μ)))
						# singleShiftQR!(S,μ,blockstart,blockend)
				# 		# if abs(e[blockend-1]) < tol*(abs(d[blockend-1]) + abs(d[blockend])) 
				# 			# blockend -= 1
				# 			# break 
				# 		# end
						# @printf("QR, blockstart: %d, blockend: %d, e[blockstart]: %e, eblockend-1]:%e, μ: %f\n", blockstart, blockend, e[blockstart], e[blockend-1], μ)
					# end
				end
				if blockstart == n break end
			end
		end
		sort!(d)
	end

	function eigvalsQL!{T<:FloatingPoint}(S::SymTridiagonal{T}; tol = eps(T), debug::Bool=false)
		d = S.dv
		e = S.ev
		n = length(d)
		blockstart = 1
		blockend = n
		@inbounds begin
			while true
				for blockend = blockstart+1:n
					if abs(e[blockend-1]) <= tol*(abs(d[blockend-1]) + abs(d[blockend]))
						blockend -= 1
						break 
					end
				end
				if blockstart == blockend
					blockstart += 1
				elseif blockstart + 1 == blockend
					d[blockstart], d[blockend] = eigvals2x2(d[blockstart],d[blockend],e[blockstart])
					blockstart += 1
				else
					# if abs(d[blockstart]) > abs(d[blockend])
						μ = (d[blockstart+1]-d[blockstart])/(2*e[blockstart])
						r = hypot(μ,one(T))
						μ = d[blockstart] - (e[blockstart]/(μ+copysign(r,μ)))
						singleShiftQL!(S,μ,blockstart,blockend)
						debug && @printf("QL, blockstart: %d, blockend: %d, e[blockstart]: %e, e[blockend-1]:%e, μ: %f\n", blockstart, blockend, e[blockstart], e[blockend-1], μ)
					# else
						# δ = 0.5*(d[blockend-1]-d[blockend])
						# μ = d[blockend] - copysign(abs2(e[blockend-1]),δ)/(abs(δ) + hypot(δeblockend-1]))
						# μ = (d[blockend-1]-d[blockend])/(2*e[blockend-1])
						# r = hypot(μ,one(T))
						# μ = d[blockend] - (e[blockend-1]/(μ+copysign(r,μ)))
						# singleShiftQR!(S,μ,blockstart,blockend)
				# 		# if abs(e[blockend-1]) < tol*(abs(d[blockend-1]) + abs(d[blockend])) 
				# 			# blockend -= 1
				# 			# break 
				# 		# end
						# @printf("QR, blockstart: %d, blockend: %d, e[blockstart]: %e, eblockend-1]:%e, μ: %f\n", blockstart, blockend, e[blockstart], e[blockend-1], μ)
					# end
				end
				if blockstart == n break end
			end
		end
		sort!(d)
	end

	function eigvalsQR!{T<:FloatingPoint}(S::SymTridiagonal{T}; tol = eps(T), debug::Bool=false)
		d = S.dv
		e = S.ev
		n = length(d)
		blockstart = 1
		blockend = n
		@inbounds begin
			while true
				for blockend = blockstart+1:n
					if abs(e[blockend-1]) <= tol*(abs(d[blockend-1]) + abs(d[blockend]))
						blockend -= 1
						break 
					end
				end
				if blockstart == blockend
					blockstart += 1
				elseif blockstart + 1 == blockend
					d[blockstart], d[blockend] = eigvals2x2(d[blockstart],d[blockend],e[blockstart])
					blockstart += 1
				else
					# δ = 0.5*(d[blockend-1]-d[blockend])
					# μ = d[blockend] - copysign(abs2(e[blockend-1]),δ)/(abs(δ) + hypot(δ,e[blockend-1]))
					μ = (d[blockend-1]-d[blockend])/(2*e[blockend-1])
					r = hypot(μ,one(T))
					μ = d[blockend] - (e[blockend-1]/(μ+copysign(r,μ)))
					singleShiftQR!(S,μ,blockstart,blockend)
				# 		# if abs(e[blockend-1]) < tol*(abs(d[blockend-1]) + abs(d[blockend])) 
				# 			# blockend -= 1
				# 			# break 
				# 		# end
					debug && @printf("QR, blockstart: %d, blockend: %d, e[blockstart]: %e, e[blockend-1]:%e, d[blockend]: %f, μ: %f\n", blockstart, blockend, e[blockstart], e[blockend-1], d[blockend], μ)
					# end
				end
				if blockstart == n break end
			end
		end
		sort!(d)
	end

	# function singleShiftQR!(S::SymTridiagonal, shift::Number, istart::Integer = 1, iend::Integer = length(S.dv))
	# 	d = S.dv
	# 	e = S.ev
	# 	n = length(d)
	# 	@inbounds begin
	# 		c, s, r = givensAlgorithm(d[istart]-shift, e[istart])
	# 		csq = c*c
	# 		ssq = s*s
	# 		d1 = d[istart]
	# 		d2 = d[istart+1]
	# 		e1 = e[istart]
	# 		d[istart] = csq*d1 + 2*c*s*e1 + ssq*d2
	# 		d[istart+1] = ssq*d1 - 2*c*s*e1 + csq*d2
	# 		e[istart] = (csq-ssq)*e1 + c*s*(d2-d1)
	# 		bulge = s*e[2]
	# 		e[istart+1] *= c
	# 		for i = istart:iend-2
	# 			c,s,r = givensAlgorithm(e[i],bulge)
	# 			csq = c*c
	# 			ssq = s*s
	# 			d1 = d[i+1]
	# 			d2 = d[i+2]
	# 			e1 = e[i+1]
	# 			d[i+1] = csq*d1 + 2*c*s*e1 + ssq*d2
	# 			d[i+2] = ssq*d1 - 2*c*s*e1 + csq*d2
	# 			e[i] = r
	# 			e[i+1] = (csq-ssq)*e1 + s*c*(d2-d1)
	# 			if i < iend-2
	# 				bulge = s*e[i+2]
	# 				e[i+2] *= c
	# 			end
	# 		end
	# 	end
	# 	S
	# end	

	function singleShiftQLPWK!(S::SymTridiagonal, shift::Number, istart::Integer = 1, iend::Integer = length(S.dv))
		d = S.dv
		e = S.ev
		n = length(d)
		γi = d[iend] - shift
		π = abs2(γi)
		ci = one(eltype(S))
		s = zero(eltype(S))
		@inbounds for i = iend-1:-1:istart
			ei = e[i]
			ζ = π + ei
			if i < iend-1 e[i+1] = s*ζ end
			ci1 = ci
			ci = π/ζ
			s = ei/ζ
			di = d[i]
			γi1 = γi
			γi = ci*(di - shift) - s*γi1
			d[i+1] = γi1 + di - γi
			π = ci == 0 ? ei*ci1 : γi*γi/ci
		end
		e[istart] = s*π
		d[istart] = shift + γi
		S
	end

	function singleShiftQL!(S::SymTridiagonal, shift::Number, istart::Integer = 1, iend::Integer = length(S.dv))
		d = S.dv
		e = S.ev
		n = length(d)
		γi = d[iend] - shift
		π = γi
		ci = one(eltype(S))
		si = zero(eltype(S))
		@inbounds for i = iend-1:-1:istart
			ei = e[i]
			ci1 = ci
			si1 = si
			ci,si,ζ = givensAlgorithm(π,ei)
			if i < iend-1 e[i+1] = si1*ζ end
			di = d[i]
			γi1 = γi
			γi = ci*ci*(di - shift) - si*si*γi1
			d[i+1] = γi1 + di - γi
			π = ci == 0 ? -ei*ci1 : γi/ci
		end
		e[istart] = si*π
		d[istart] = shift + γi
		S
	end

	function singleShiftQR!(S::SymTridiagonal, shift::Number, istart::Integer = 1, iend::Integer = length(S.dv))
		d = S.dv
		e = S.ev
		n = length(d)
		γi = d[istart] - shift
		π = γi
		ci = one(eltype(S))
		si = zero(eltype(S))
		@inbounds for i = istart+1:iend
			ei = e[i]
			ci1 = ci
			si1 = si
			ci,si,ζ = givensAlgorithm(π,ei)
			if i > istart e[i-1] = si1*ζ end
			di = d[i-1]
			γi1 = γi
			γi = ci*ci*(di - shift) - si*si*γi1
			d[i] = γi1 + di - γi
			π = ci == 0 ? -ei*ci1 : γi/ci
		end
		e[iend-1] = si*π
		d[iend] = shift + γi
		S
	end

	function singleShiftQRv2!(S::SymTridiagonal, shift::Number, istart::Integer = 1, iend::Integer = length(S.dv))
		d = S.dv
		e = S.ev
		n = length(d)
		@inbounds begin
			G = givens(d[istart]-shift, e[istart],istart,istart+1,n)
			c, s, r = G.c, G.s, G.r
			csq = c*c
			ssq = abs2(s)
			d1 = d[istart]
			d2 = d[istart+1]
			e1 = e[istart]
			d[istart] = csq*d1 + 2*c*real(s*e1) + ssq*d2
			d[istart+1] = ssq*d1 - 2*c*real(conj(s)*e1) + csq*d2
			e[istart] = c*(conj(s)*d2 - s*d1) + 2*csq*real(e1) - conj(e1)
			bulge = conj(s)*e[2]
			e[istart+1] *= c
			for i = istart:iend-2
				G = givens(e[i],bulge,i,i+1,n)
				c, s, r = G.c, G.s, G.r
				csq = c*c
				ssq = abs2(s)
				d1 = d[i+1]
				d2 = d[i+2]
				e1 = e[i+1]
				d[i+1] = csq*d1 + 2*c*real(s*e1) + ssq*d2
				d[i+2] = ssq*d1 - 2*c*real(conj(s)*e1) + csq*d2
				e[i] = r
				e[i+1] = c*(conj(s)*d2 - s*d1) + 2*csq*real(e1) - conj(e1)
				if i < iend-2
					bulge = conj(s)*e[i+2]
					e[i+2] *= c
				end
			end
		end
		S
	end

	function zeroshiftQR!{T}(A::Bidiagonal{T})
		d = A.dv
		e = A.ev
		n = length(d)
		oldc = one(T)
		olds = oldc
		c = oldc
		for i = 1:n-1
			c, s, r = givensAlgorithm(d[i]*c,e[i])
			if i > 1 e[i-1] = olds*r end
			oldc, olds, d[i] = givensAlgorithm(oldc*r,d[i+1]*s)
		end
		h = d[n]*c
		e[n-1] = h*olds
		d[n] = h*oldc
		return A
	end

	# immutable BidiagonalFactorization{T,A<:AbstractMatrix{T}} <: Factorization{T}
		# matrix::A
		# rotationLeft::Rotation{T}
		# rotationRigth::Rotation{T}
	# end

	# function singleShiftQR!(S::SymTridiagonal, shift::Number, m::Integer = length(S.S.dv))
	# 	d = S.dv
	# 	e = S.ev
	# 	n = length(d)
	# 	G = givens(d[1]-shift, e[1], 1, 2, n)
	# 	csq = G.c*G.c
	# 	ssq = abs2(G.s)
	# 	d1 = d[1]
	# 	d2 = d[2]
	# 	e1 = e[1]
	# 	d[1] = csq*d1 + 2*G.c*real(G.s*e1) + ssq*d2
	# 	d[2] = ssq*d1 - 2*G.c*real(conj(G.s)*e1) + csq*d2
	# 	e[1] = G.c*(conj(G.s)*d2 - G.s*d1) + 2*csq*real(e1) - conj(e1)
	# 	bulge = conj(G.s)*e[2]
	# 	e[2] *= G.c
	# 	for i = 1:m-2
	# 		G = givens(e[i],bulge,i+1,i+2,n)
	# 		csq = G.c*G.c
	# 		ssq = abs2(G.s)
	# 		d1 = d[i+1]
	# 		d2 = d[i+2]
	# 		e1 = e[i+1]
	# 		d[i+1] = csq*d1 + 2*G.c*real(G.s*e1) + ssq*d2
	# 		d[i+2] = ssq*d1 - 2*G.c*real(conj(G.s)*e1) + csq*d2
	# 		e[i] = G.r
	# 		e[i+1] = G.c*(conj(G.s)*d2 - G.s*d1) + 2*csq*real(e1) - conj(e1)
	# 		if i < m-2
	# 			bulge = conj(G.s)*e[i+2]
	# 			e[i+2] *= G.c
	# 		end
	# 	end
	# 	S
	# end
	function mysymtri!{T}(A::AbstractMatrix{T})
		n = chksquare(A)
		τ = Array(T, n-2)
		for k = 1:n-2
			τk = elementary!(A, k+1, k)
			τ[k] = τk
			for j = k+1:n
				νAj = A[k+1,j]
				for i = k+2:n
					νAj += conj(A[i,k])*A[i,j]
				end

				νAj *= τk
				A[k+1,j] -= νAj
				for i = k+2:n
					A[i,j] -= A[i,k]*νAj
				end

				νAj = A[k+1,j]
				for i = k+2:n
					νAj += conj(A[i,k])*A[i,j]
				end
			end	
		end
		A
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

	eigvals!(A::SymmetricTridiagonalFactorization) = eigvals(A.diagonals)
	eigvals!(A::Hermitian) = eigvals!(symtri!(A))


end