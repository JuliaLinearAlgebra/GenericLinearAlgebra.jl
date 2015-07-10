module LAPACK2

    using Base.LinAlg: BlasInt, chkstride1, LAPACKException
    using Base.LinAlg.LAPACK: @lapackerror

    # LAPACK wrappers

    ## Standard QR/QL
    function steqr!(compz::Char, d::StridedVector{Float64}, e::StridedVector{Float64}, Z::StridedMatrix{Float64}, work::StridedVector{Float64} = compz == 'N' ? Array(Float64, 0) : Array(Float64, max(1, 2n-2)))

        # Extract sizes
        n = length(d)
        ldz = stride(Z, 2)

        # Checks
        length(e) + 1 == n || throw(DimensionMismatch(""))
        if compz == 'N'
        elseif compz in ('V', 'I')
            chkstride1(Z)
            n <= size(Z, 1) || throw(DimensionMismatch("Z matrix has too few rows"))
            n <= size(Z, 2) || throw(DimensionMismatch("Z matrix has too few colums"))
        else
            throw(ArgumentError("compz must be either 'N', 'V', or 'I'"))
        end

        # Allocations
        info = Array(BlasInt,1)

        ccall(("dsteqr_", Base.liblapack_name),Void,
            (Ptr{Uint8}, Ptr{BlasInt}, Ptr{Float64}, Ptr{Float64},
             Ptr{Float64}, Ptr{BlasInt}, Ptr{Float64}, Ptr{BlasInt}),
             &compz, &n, d, e,
             Z, &max(1, ldz), work, info)
        info[1] == 0 || throw(LAPACKException(info[1]))

        return d, Z
    end

    ## Square root free QR/QL (values only, but fast)
    function sterf!(d::StridedVector{Float64}, e::StridedVector{Float64})

        # Extract sizes
        n = length(d)

        # Checks
        length(e) >= n - 1 || throw(DimensionMismatch("subdiagonal is too short"))

        # Allocations
        info = BlasInt[0]

        ccall((:dsterf_, Base.liblapack_name), Void,
            (Ptr{BlasInt}, Ptr{Float64}, Ptr{Float64}, Ptr{BlasInt}),
            &n, d, e, info)

        info[1] == 0 || throw(LAPACKException(info[1]))

        d
    end

    ## Divide and Conquer
    function stedc!(compz::Char, d::StridedVector{Float64}, e::StridedVector{Float64}, Z::StridedMatrix{Float64}, work::StridedVector{Float64}, lwork::BlasInt, iwork::StridedVector{BlasInt}, liwork::BlasInt)

        # Extract sizes
        n = length(d)
        ldz = stride(Z, 2)

        # Checks
        length(e) + 1 == n || throw(DimensionMismatch(""))
        # length(work)
        # length(lwork)

        # Allocations
        info = BlasInt[0]

        ccall(("dstedc_", Base.liblapack_name), Void,
                (Ptr{Uint8}, Ptr{BlasInt}, Ptr{Float64}, Ptr{Float64},
                 Ptr{Float64}, Ptr{BlasInt}, Ptr{Float64}, Ptr{BlasInt},
                 Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                &compz, &n, d, e,
                Z, &max(1, ldz), work, &lwork,
                iwork, &liwork, info)

        info[1] == 0 || throw(LAPACKException(info[1]))

        return d, Z
    end

    function stedc!(compz::Char, d::StridedVector{Float64}, e::StridedVector{Float64}, Z::StridedMatrix{Float64})

        work::Vector{Float64} = Float64[0]
        iwork::Vector{BlasInt} = BlasInt[0]
        stedc!(compz, d, e, Z, work, -1, iwork, -1)

        lwork::BlasInt = work[1]
        work = Array(Float64, lwork)
        liwork = iwork[1]
        iwork = Array(BlasInt, liwork)

        return stedc!(compz, d, e, Z, work, lwork, iwork, liwork)
    end

    ## RRR
    for (lsymb, elty) in ((:dstemr_, :Float64), (:sstemr_, :Float32))
        @eval begin
            function stemr!(jobz::Char, range::Char, dv::StridedVector{$elty}, ev::StridedVector{$elty}, vl::$elty, vu::$elty, il::BlasInt, iu::BlasInt, w::StridedVector{$elty}, Z::StridedMatrix{$elty}, nzc::BlasInt, isuppz::StridedVector{BlasInt}, work::StridedVector{$elty}, lwork::BlasInt, iwork::StridedVector{BlasInt}, liwork::BlasInt)

                # Extract sizes
                n = length(dv)
                ldz = stride(Z, 2)

                # Checks
                length(ev) >= n - 1 || throw(DimensionMismatch("subdiagonal is too short"))

                # Allocations
                eev::Vector{$elty} = length(ev) == n - 1 ? [ev, zero($elty)] : copy(ev)
                abstol = Array($elty, 1)
                m = Array(BlasInt, 1)
                tryrac = BlasInt[1]
                info = Array(BlasInt, 1)
                ccall(($(string(lsymb)), Base.liblapack_name), Void,
                    (Ptr{Char}, Ptr{Char}, Ptr{BlasInt}, Ptr{$elty},
                    Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt},
                    Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty},
                    Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                    Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                    Ptr{BlasInt}),
                    &jobz, &range, &n, dv,
                    eev, &vl, &vu, &il,
                    &iu, m, w, Z,
                    &ldz, &nzc, isuppz, tryrac,
                    work, &lwork, iwork, &liwork,
                    info)

                info[1] == 0 || throw(LAPACKException(info[1]))

                w, Z, tryrac[1]
            end

            function stemr!(jobz::Char, range::Char, dv::StridedVector{$elty}, ev::StridedVector{$elty}, vl::$elty = typemin($elty), vu::$elty = typemax($elty), il::BlasInt = 1, iu::BlasInt = length(dv))
                n = length(dv)
                w = Array($elty, n)
                if jobz == 'N'
                    # Z = Array($elty, 0, 0)
                    nzc::BlasInt = 0
                    isuppz = Array(BlasInt, 0)
                elseif jobz == 'V'
                    nzc = -1
                    isuppz = similar(dv, BlasInt, 2n)
                else
                    throw(ArgumentError("jobz must be either 'N' or 'V'"))
                end

                work = Array($elty, 1)
                lwork::BlasInt = -1
                iwork = BlasInt[0]
                liwork::BlasInt = -1
                Z = Array($elty, 1, 1)
                nzc = -1

                stemr!(jobz, range, dv, ev, vl, vu, il, iu, w, Z, nzc, isuppz, work, lwork, iwork, liwork)

                lwork = work[1]
                work = Array($elty, lwork)
                liwork = iwork[1]
                iwork = Array(BlasInt, liwork)
                nzc = Z[1]
                Z = similar(dv, $elty, n, nzc)

                return stemr!(jobz, range, dv, ev, vl, vu, il, iu, w, Z, nzc, isuppz, work, lwork, iwork, liwork)
            end
        end
    end

    ## Cholesky + singular values
    function spteqr!(compz::Char, d::StridedVector{Float64}, e::StridedVector{Float64}, Z::StridedMatrix{Float64}, work::StridedVector{Float64} = Array(Float64, 4length(d)))

        n = length(d)
        ldz = stride(Z, 2)

        # Checks
        length(e) >= n - 1 || throw(DimensionMismatch("subdiagonal vector is too short"))
        if compz == 'N'
        elseif compz == in('V', 'I')
            size(Z, 1) >= n     || throw(DimensionMismatch("Z does not have enough rows"))
            size(Z, 2) >= ldz   || throw(DimensionMismatch("Z does not have enough columns"))
        else
            throw(ArgumentError("compz must be either 'N', 'V', or 'I'"))
        end

        info = Array(BlasInt, 1)

        ccall((:dpteqr_, Base.liblapack_name), Void,
            (Ptr{Char}, Ptr{BlasInt}, Ptr{Float64}, Ptr{Float64},
             Ptr{Float64}, Ptr{BlasInt}, Ptr{Float64}, Ptr{BlasInt}),
            &compz, &n, d, e,
            Z, &max(1, ldz), work, info)

        info[1] == 0 || throw(LAPACKException(info[1]))

        d, Z
    end



end