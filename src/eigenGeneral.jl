# Eigenfunctionality

module EigenGeneral

	import Base: A_mul_B!, A_mul_Bc!, Ac_mul_B, A_mul_Bc, A_ldiv_B!, ctranspose, full, getindex, size
	import Base.LinAlg: chksquare, QRPackedQ, Rotation

	type Hessenberg{T} <: Factorization{T}
		H::Matrix{T}
		R::Base.LinAlg.Rotation{Base.LinAlg.Givens{T}}
	end

	type Schur{T} <: Factorization{T}
		S::Matrix{T}
		R::Base.LinAlg.Rotation{Base.LinAlg.Givens{T}}
	end

	# getindex{T}(H::Hessenberg{T}, i::Integer, j::Integer) = i > j+1 ? zero(T) : getindex(H.H, i, j)
	size(H::Hessenberg, args...) = size(H.H, args...)

	# type Householder{T}
	# 	h::Vector{T}
	# 	τ::T
	# end

	function adiagmax(A::AbstractMatrix)
		adm = zero(typeof(real(A[1])))
		@inbounds begin
			for i = size(A,1)
				adm = max(adm, abs(A[i,i]))
			end
		end
		return adm
	end

	function eigvals!{T}(A::AbstractMatrix{T}; tol = eps(adiagmax(A))*size(A,1)^3)
		n = chksquare(A)
		H = hessenberg!(A)
		schur!(H, tol = tol)
		vals = Array(Complex{T}, n)
		i = 1
		while i < n
			Hii = H.H[i,i]
			Hi1i1 = H.H[i+1,i+1]
			if abs(H.H[i+1,i]) < tol*(abs(Hi1i1) + abs(Hii))
				vals[i] = Hii
				i += 1
			else
				d = Hii*Hi1i1 - H.H[i,i+1]*H.H[i+1,i]
				t = Hii + Hi1i1
				x = 0.5*t
				if d < x*x println(i) end
				y = sqrt(d - x*x)
				vals[i] = complex(x,y)
				vals[i+1] = complex(x,-y)
				i += 2
			end
		end
		if i == n vals[i] = H.H[n,n] end
		return vals
	end

	function hessenberg!{T}(A::AbstractMatrix{T})
		n = chksquare(A)
		R = Base.LinAlg.Rotation(Base.LinAlg.Givens{T}[])
		for j = 1:n-2
			for i = j+2:n
				G = givens(A,j+1,i,j)
				A_mul_B!(G,A)
				A_mul_Bc!(A,G)
				A_mul_B!(G,R)
			end
		end
		return Hessenberg(A, R)
	end		
	
	function schur!{T<:Real}(H::Hessenberg{T}; tol = eps(adiagmax(H.H))*size(H,1)^3)
		n = size(H, 1)
		m = n
		while m > 1
			Hmm = H.H[m,m]
			Hm1m1 = H.H[m-1,m-1]
			d = Hm1m1*Hmm-H.H[m,m-1]H.H[m-1,m]
			t = Hm1m1 + Hmm
			if t*t > 4d
				singleShiftQR!(H, 0.5*(t + sqrt(t*t - 4d)), m)
				if abs(H.H[m,m-1]) < tol*(abs(Hm1m1)+abs(Hmm))
					m -= 1
				end
			elseif m == 2
				break
			else
				doubleShiftQR!(H, t, d, m)
				if abs(H.H[m,m-1]) < tol*(abs(Hm1m1)+abs(Hmm))
					m -= 1
				elseif abs(H.H[m-1,m-2]) < tol*(abs(H.H[m-2,m-2])+abs(Hm1m1))
					m -= 2
				end
			end
		end
		return Schur(H.H,H.R)
	end

	function doubleShiftQR!(H::Hessenberg, shiftTrace::Number, shiftDeterminant::Number, m::Integer = size(H, 1))
		sH1 = size(H, 1)
		H11 = H.H[1,1]
		H21 = H.H[2,1]
		G1 = givens(H11*H11 + H.H[1,2]*H21 - shiftTrace*H11 + shiftDeterminant, H21*(H11 + H.H[2,2] - shiftTrace),1,2,sH1)
		G2 = givens(G1.r, H21*H.H[3,2],1,3,sH1)
		A_mul_B!(G1,H.H)
		A_mul_B!(G2,H.H)
		A_mul_Bc!(H.H,G1)
		A_mul_Bc!(H.H,G2)
		A_mul_B!(G1,H.R)
		A_mul_B!(G2,H.R)
		for i = 1:m-2
			for j = 1:2
				if i+j+1>m break end
				# G = givens(H.H,i+1,i+j+1,i)
				G = givens(H.H[i+1,i],H.H[i+j+1,i],i+1,i+j+1,sH1)
				A_mul_B!(G,H.H)
				A_mul_Bc!(H.H,G)
				A_mul_B!(G,H.R)
			end
		end
		return H
	end
end