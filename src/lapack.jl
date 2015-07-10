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

### Rectangular full packed format

# Symmetric rank-k operation for matrix in RFP format.
for (fn, elty, relty) in ((:dsfrk_, :Float64, :Float64),
                   (:ssfrk_, :Float32, :Float32),
                   (:zhfrk_, :Complex128, :Float64),
                   (:chfrk_, :Complex64, :Float32))
    @eval begin
        function sfrk!(transr::Char, uplo::Char, trans::Char, alpha::Real, A::StridedMatrix{$elty}, beta::Real, C::StridedVector{$elty})
            chkuplo(uplo)
            chkstride1(A)
            if trans == 'N' || trans == 'n'
                n, k = size(A)
            elseif trans == 'T' || trans == 't'
                k, n = size(A)
            end
            lda = max(1, stride(A, 2))
            ccall(($(blasfunc(fn)), liblapack), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt},
                 Ptr{BlasInt}, Ptr{$relty}, Ptr{$elty}, Ptr{BlasInt},
                 Ptr{$relty}, Ptr{$elty}),
                &transr, &uplo, &trans, &n,
                &k, &alpha, A, &lda,
                &beta, C)
            C
        end
    end
end

# Cholesky factorization of a real symmetric positive definite matrix A
for (fn, elty) in ((:dpftrf_, :Float64),
                   (:spftrf_, :Float32),
                   (:zpftrf_, :Complex128),
                   (:cpftrf_, :Complex64))
    @eval begin
        function pftrf!(transr::Char, uplo::Char, A::StridedVector{$elty})
            chkuplo(uplo)
            n = round(Int,div(sqrt(8length(A)), 2))
            info = Array(BlasInt, 1)
            ccall(($(blasfunc(fn)), liblapack), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{BlasInt}),
                &transr, &uplo, &n, A,
                info)
            @assertargsok
            @assertnonsingular
            A
        end
    end
end

# Computes the inverse of a (real) symmetric positive definite matrix A using the Cholesky factorization
for (fn, elty) in ((:dpftri_, :Float64),
                   (:spftri_, :Float32),
                   (:zpftri_, :Complex128),
                   (:cpftri_, :Complex64))
    @eval begin
        function pftri!(transr::Char, uplo::Char, A::StridedVector{$elty})
            chkuplo(uplo)
            n = round(Int,div(sqrt(8length(A)), 2))
            info = Array(BlasInt, 1)
            ccall(($(blasfunc(fn)), liblapack), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{BlasInt}),
                &transr, &uplo, &n, A,
                info)
            @assertargsok
            @assertnonsingular
            A
        end
    end
end

# DPFTRS solves a system of linear equations A*X = B with a symmetric positive definite matrix A using the Cholesky factorization
for (fn, elty) in ((:dpftrs_, :Float64),
                   (:spftrs_, :Float32),
                   (:zpftrs_, :Complex128),
                   (:cpftrs_, :Complex64))
    @eval begin
        function pftrs!(transr::Char, uplo::Char, A::StridedVector{$elty}, B::StridedVecOrMat{$elty})
            chkuplo(uplo)
            chkstride1(B)
            n = round(Int,div(sqrt(8length(A)), 2))
            if n != size(B, 1)
                throw(DimensionMismatch("B has first dimension $(size(B,1)) but needs $n"))
            end
            nhrs = size(B, 2)
            ldb = max(1, stride(B, 2))
            info = Array(BlasInt, 1)
            ccall(($(blasfunc(fn)), liblapack), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                &transr, &uplo, &n, &nhrs,
                A, B, &ldb, info)
            @assertargsok
            @assertposdef
            B
        end
    end
end

# Solves a matrix equation (one operand is a triangular matrix in RFP format)
for (fn, elty) in ((:dtfsm_, :Float64),
                   (:stfsm_, :Float32),
                   (:ztfsm_, :Complex128),
                   (:ctfsm_, :Complex64))
    @eval begin
        function pftrs!(transr::Char, side::Char, uplo::Char, trans::Char, diag::Char, alpha::Real, A::StridedVector{$elty}, B::StridedMatrix{$elty})
            chkuplo(uplo)
            chkside(side)
            chkdiag(diag)
            chkstride1(B)
            m, n = size(B)
            if round(Int, div(sqrt(8length(A)), 2)) != m
                throw(DimensionMismatch("First dimension of B must equal $(round(Int, div(sqrt(8length(A)), 2))), got $m"))
            end
            ldb = max(1, stride(B, 2))
            ccall(($(blasfunc(fn)), liblapack), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8},
                 Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}),
                &transr, &side, &uplo, &trans,
                &diag, &m, &n, &alpha,
                A, B, &ldb)
            B
        end
    end
end

# Computes the inverse of a triangular matrix A stored in RFP format.
for (fn, elty) in ((:dtftri_, :Float64),
                   (:stftri_, :Float32),
                   (:ztftri_, :Complex128),
                   (:ctftri_, :Complex64))
    @eval begin
        function tftri!(transr::Char, uplo::Char, diag::Char, A::StridedVector{$elty})
            chkuplo(uplo)
            chkdiag(diag)
            n = round(Int,div(sqrt(8length(A)), 2))
            info = Array(BlasInt, 1)
            ccall(($(blasfunc(fn)), liblapack), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{BlasInt}),
                &transr, &uplo, &diag, &n,
                A, info)
            @assertargsok
            @assertnonsingular
            A
        end
    end
end

# Copies a triangular matrix from the rectangular full packed format (TF) to the standard full format (TR)
for (fn, elty) in ((:dtfttr_, :Float64),
                   (:stfttr_, :Float32),
                   (:ztfttr_, :Complex128),
                   (:ctfttr_, :Complex64))
    @eval begin
        function tfttr!(transr::Char, uplo::Char, Arf::StridedVector{$elty})
            chkuplo(uplo)
            n = round(Int,div(sqrt(8length(Arf)), 2))
            info = Array(BlasInt, 1)
            A = similar(Arf, $elty, n, n)
            ccall(($(blasfunc(fn)), liblapack), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                &transr, &uplo, &n, Arf,
                A, &n, info)
            @assertargsok
            A
        end
    end
end

# Copies a triangular matrix from the standard full format (TR) to the rectangular full packed format (TF).
for (fn, elty) in ((:dtrttf_, :Float64),
                   (:strttf_, :Float32),
                   (:ztrttf_, :Complex128),
                   (:ctrttf_, :Complex64))
    @eval begin
        function trttf!(transr::Char, uplo::Char, A::StridedMatrix{$elty})
            chkuplo(uplo)
            chkstride1(A)
            n = size(A, 1)
            lda = max(1, stride(A, 2))
            info = Array(BlasInt, 1)
            Arf = similar(A, $elty, div(n*(n+1), 2))
            ccall(($(blasfunc(fn)), liblapack), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}),
                &transr, &uplo, &n, A,
                &lda, Arf, info)
            @assertargsok
            Arf
        end
    end
end

end