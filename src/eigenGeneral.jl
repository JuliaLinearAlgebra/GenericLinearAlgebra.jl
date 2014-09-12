# Eigenfunctionality

module EigenGeneral

	using ..HouseholderModule: Householder, householder!
	using Base.LinAlg: Givens, Rotation, chksquare
	using ArrayViews

	import Base: A_mul_B!, A_mul_Bc!, Ac_mul_B, A_mul_Bc, A_ldiv_B!, ctranspose, full, getindex, size
	import Base.LinAlg: QRPackedQ 

	# Auxiliary
	function adiagmax(A::AbstractMatrix)
		adm = zero(typeof(real(A[1])))
		@inbounds begin
			for i = size(A,1)
				adm = max(adm, abs(A[i,i]))
			end
		end
		return adm
	end

	# Hessenberg Matrix
	immutable HessenbergMatrix{T,S<:AbstractMatrix{T}} <: AbstractMatrix{T}
		data::S
	end

	copy{T,S}(H::HessenbergMatrix{T,S}) = HessenbergMatrix{T,S}(copy(H.data))

	getindex{T,S}(H::HessenbergMatrix{T,S}, i::Integer, j::Integer) = i > j + 1 ? zero(T) : H.data[i,j]

	size(H::HessenbergMatrix) = size(H.data)
	size(H::HessenbergMatrix, i::Integer) = size(H.data, i)

	function A_ldiv_B!(H::HessenbergMatrix, B::AbstractVecOrMat)
		n = size(H, 1)
		Hd = H.data
		for i = 1:n-1
			G = givens!(Hd, i, i+1, i)
			A_mul_B!(G, view(Hd, 1:n, i+1:n))
			A_mul_B!(G, B)
		end
		A_ldiv_B!(Triangular(Hd, :U), B)
	end

	# Hessenberg factorization
	immutable HessenbergFactorization{T, S<:DenseMatrix{T},U} <: Factorization{T}
		data::S
		τ::Vector{U}
	end

	function hessfact!{T}(A::AbstractMatrix{T})
		n = chksquare(A)
		τ = Array(Householder{T}, n - 1)
		for i = 1:n - 1
			H = householder!(view(A, i + 1, i), view(A, i + 2:n, i))
			τ[i] = H
			Ac_mul_B!(H, view(A, i + 1:n, i + 1:n))
			A_mul_B!(view(A, :, i + 1:n), H)
		end
		return HessenbergFactorization{T, typeof(A), eltype(τ)}(A, τ)
	end	

	size(H::HessenbergFactorization, args...) = size(H.data, args...)

	# Schur
	immutable Schur{T,S<:DenseMatrix{T}} <: Factorization{T}
		data::S
		R::Rotation
	end
	
	function schurfact!{T<:Real}(H::HessenbergFactorization{T}; tol = eps(adiagmax(H.data))*size(H,1)^3)
		n = size(H, 1)
		m = n
		HH = H.data
		τ = Rotation(Givens{T}[])
		while m > 1
			println("m = ", m)
			Hmm = HH[m, m]
			Hm1m1 = HH[m - 1, m - 1]
			d = Hm1m1*Hmm - HH[m, m - 1]*HH[m - 1, m]
			t = Hm1m1 + Hmm
			if t*t > 4d
				println("Single shift!")
				singleShiftQR!(H, τ, 0.5*(t + sqrt(t*t - 4d)), m)
				if abs(HH[m, m - 1]) < tol*(abs(Hm1m1) + abs(Hmm))
					m -= 1
				end
			elseif m == 2
				break
			else
				println("Double shift!")
				doubleShiftQR!(H, τ, t, d, m)
				if abs(HH[m, m - 1]) < tol*(abs(Hm1m1) + abs(Hmm))
					m -= 1
				elseif abs(HH[m - 1, m - 2]) < tol*(abs(HH[m - 2, m - 2]) + abs(Hm1m1))
					m -= 2
				end
			end
		end
		return Schur{T,typeof(HH)}(HH, τ)
	end
	schurfact!(A::DenseMatrix; tol = eps(adiagmax(A))*size(A,1)^3) = schurfact!(hessfact!(A), tol = tol)

	function singleShiftQR!(H::HessenbergFactorization, τ::Rotation, shift::Number, m::Integer = size(H, 1))
		sH1 = size(H, 1)
		HH = H.data
		H11 = HH[1, 1]
		H21 = HH[2, 1]
		if m > 2
			Htmp = HH[3, 1]
			HH[3, 1] = 0
		end
		G = givens(H11 - shift, H21, 1, 2, sH1)
		A_mul_B!(G, HH)
		A_mul_Bc!(view(HH, 1:min(3, m), :), G)
		A_mul_B!(G, τ)
		for i = 1:m - 2
			G = givens(HH[i + 1, i], HH[i + 2, i], i + 1, i + 2, sH1)
			A_mul_B!(G, view(HH, :, i:m))
			HH[i + 2, i] = Htmp
			if i < m - 2
				Htmp = HH[i + 3, i + 1]
				HH[i + 3, i + 1] = 0
			end
			A_mul_Bc!(view(HH, 1:min(i + 3, m), :), G)
			A_mul_B!(G, τ)
		end
		return H
	end

	function doubleShiftQR!(H::HessenbergFactorization, τ::Rotation, shiftTrace::Number, shiftDeterminant::Number, m::Integer = size(H, 1))
		sH1 = size(H, 1)
		HH = H.data
		H11 = HH[1, 1]
		H21 = HH[2, 1]
		Htmp11 = HH[3, 1]
		HH[3, 1] = 0
		Htmp21 = HH[4, 1]
		HH[4, 1] = 0
		Htmp22 = HH[4, 2]
		HH[4, 2] = 0
		G1 = givens(H11*H11 + HH[1, 2]*H21 - shiftTrace*H11 + shiftDeterminant, H21*(H11 + HH[2, 2] - shiftTrace), 1, 2, sH1)
		G2 = givens(G1.r, H21*HH[3, 2], 1, 3, sH1)
		A_mul_B!(G1, HH)
		A_mul_B!(G2, HH)
		A_mul_Bc!(view(HH, 1:4, :), G1)
		A_mul_Bc!(view(HH, 1:4, :), G2)
		A_mul_B!(G1, τ)
		A_mul_B!(G2, τ)
		for i = 1:m - 2
			for j = 1:2
				if i + j + 1 > m break end
				# G = givens(H.H,i+1,i+j+1,i)
				G = givens(HH[i + 1, i], HH[i + j + 1, i], i + 1, i + j + 1, sH1)
				A_mul_B!(G, view(HH, :, i:m))
				HH[i + j + 1, i] = Htmp11
				Htmp11 = Htmp21
				Htmp21 = Htmp22
				if i + 4 <= m
					Htmp22 = HH[i + 4, i + j]
					HH[i + 4, i + j] = 0
				end
				A_mul_Bc!(view(HH, 1:min(i + j + 2, m), :), G)
				A_mul_B!(G, τ)
			end
		end
		return H
	end

	eigvals!(A::AbstractMatrix; tol = eps(adiagmax(A))*size(A,1)^3) = eigvals!(schurfact!(A, tol = tol))
	eigvals!(H::HessenbergMatrix; tol = eps(adiagmax(H.data))*size(H.data,1)^3) = eigvals!(schurfact!(H, tol = tol))
	eigvals!(H::HessenbergFactorization; tol = eps(adiagmax(H.data))*size(H.data, 1)^3) = eigvals!(schurfact!(H, tol = tol))

	function eigvals!{T}(S::Schur{T}; tol = eps(adiagmax(S.data))*size(S.data,1)^3)
		HH = S.data
		n = size(HH, 1)
		vals = Array(Complex{T}, n)
		i = 1
		while i < n
			Hii = HH[i, i]
			Hi1i1 = HH[i + 1, i + 1]
			if abs(HH[i + 1, i]) < tol*(abs(Hi1i1) + abs(Hii))
				vals[i] = Hii
				i += 1
			else
				d = Hii*Hi1i1 - HH[i, i + 1]*HH[i + 1, i]
				t = Hii + Hi1i1
				x = 0.5*t
				if d < x*x println(d - x*x, i) end
				y = sqrt(d - x*x)
				vals[i] = complex(x, y)
				vals[i + 1] = complex(x, -y)
				i += 2
			end
		end
		if i == n vals[i] = HH[n, n] end
		return vals
	end	
end