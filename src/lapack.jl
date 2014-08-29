module LAPACK2
	
	using Base.LinAlg.BlasInt
	
	function steqr!(compz::Char, d::StridedVector{Float64}, e::StridedVector{Float64}, Z::StridedMatrix{Float64})
		n = length(d)
		length(e) + 1 == n || throw(DimensionMismatch(""))
		ldz = max(1,stride(Z,1))
		work = Array(Float64, max(1,2n-2))
		info = Array(BlasInt,1)
		ccall(("dsteqr_", Base.liblapack_name),Void,
			(Ptr{Uint8}, Ptr{BlasInt}, Ptr{Float64}, Ptr{Float64},
			 Ptr{Float64}, Ptr{BlasInt}, Ptr{Float64}, Ptr{BlasInt}),
	 		&compz, &n, d, e, 
	 		Z, &ldz, work, info)
		return d, Z, info[1]
	end

	function stedc!(compz::Char, d::StridedVector{Float64}, e::StridedVector{Float64}, Z::StridedMatrix{Float64})
		n = length(d)
		ldz = max(1,stride(Z,2))
		work = Float64[0]
		lwork::BlasInt = -1
		iwork = BlasInt[0]
		liwork::BlasInt = -1
		info = BlasInt[0]
		length(e) + 1 == n || throw(DimensionMismatch(""))
		for i = 1:2
			ccall(("dstedc_", Base.liblapack_name), Void,
				(Ptr{Uint8}, Ptr{BlasInt}, Ptr{Float64}, Ptr{Float64},
				 Ptr{Float64}, Ptr{BlasInt}, Ptr{Float64}, Ptr{BlasInt},
				 Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
				&compz, &n, d, e, 
				Z, &ldz, work, &lwork, 
				iwork, &liwork, info)
			if i == 1
				lwork = work[1]
				work = Array(Float64,lwork)
				liwork = iwork[1]
				iwork = Array(BlasInt,liwork)
			end
		end
		return d, Z, info[1]
	end
end