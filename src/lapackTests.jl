module LAPACKTests

const LAPACK_ROW_MAJOR = 101
const LAPACK_COL_MAJOR = 102

using Base.LinAlg: BlasInt

using ..SVDModule

function latmr!(dist::Char, iseed::Vector{BlasInt}, sym::Char, d::Vector{Float64},
    mode::BlasInt, cond::Float64, dmax::Float64, rsign::Char, grade::Char,
    dl::Vector{Float64}, model::BlasInt, condl::Float64, dr::Vector{Float64},
    moder::BlasInt, condr::Float64, pivtng::Char, ipivot::Vector{BlasInt},
    kl::BlasInt, ku::BlasInt, sparse::Float64, anorm::Float64, pack::Char,
    A::StridedMatrix{Float64})

    m, n = size(A)
    lda = stride(A, 2)

    if dist != 'U' && dist != 'S' && dist != 'N'
        throw(ArgumentError("dist must be either 'U', 'S', or 'N'"))
    end
    if length(iseed) != 4
        throw(DimensionMismatch("iseed must have length 4 but had length $(length(iseed))"))
    end
    if sym != 'S' && sym != 'H' && sym != 'N'
        throw(ArgumentError("sym must be either 'S', 'H', or 'N'"))
    end
    if length(d) != min(m,n)
        throw(DimensionMismatch("diagonal vector must have length $(min(m,n)) but had length $(length(d))"))
    end
    if abs(mode) > 6
        throw(ArgumentError("mode must be between -6 and 6 but was $mode"))
    end
    if cond < 1 && 0 < abs(mode) < 6
        throw(ArgumentError("cond must be larger than one but had value $cond"))
    end
    if rsign != 'T' && rsign != 'F' && 0 < abs(mode) < 6
        throw(ArgumentError("rsign must be either 'T' or 'F' but was $rsign"))
    end
    if grade != 'N' && grade != 'L' && grade != 'R' && grade != 'B' &&
        grade != 'S' && grade != 'H' && grade != 'E'
        throw(ArgumentError("grade must be either 'N', 'L', 'R', 'B', 'S', 'H', or 'E'"))
    end
    if length(dl) != m
        throw(DimensionMismatch("dl vector must have length $m but had length $(length(dl))"))
    end
    if abs(model) > 6
        throw(ArgumentError("model must be between -6 and 6 but was $model"))
    end
    if condl < 1 && 0 < abs(model) < 6
        throw(ArgumentError("condl must be larger than one but had value $condl"))
    end
    if length(dr) != n
        throw(DimensionMismatch("dl vector must have length $n but had length $(length(dl))"))
    end
    if abs(moder) > 6
        throw(ArgumentError("moder must be between -6 and 6 but was $moder"))
    end
    if condr < 1 && 0 < abs(moder) < 6
        throw(ArgumentError("condr must be larger than one but had value $condr"))
    end
    if pivtng == 'L'
        if length(ipivot) != m
            throw(DimensionMismatch("ipivot must have length $m but had length $(length(ipivot))"))
        end
    elseif pivtng == 'R'
        if length(ipivot) != n
            throw(DimensionMismatch("ipivot must have length $n but had length $(length(ipivot))"))
        end
    elseif pivtng == 'B' || pivtng == 'F'
        if m != n
            throw(DimensionMismatch("matrix must be square when pivtng is $pivtng"))
        end
        if length(pivtng) != m
            throw(DimensionMismatch("ipivot must have length $m but had length $(length(ipivot))"))
        end
    elseif pivtng != 'N'
        throw(ArgumentError("pivtng must be either 'N', 'L', 'R', 'B', or 'F' but was &pivtng"))
    end

    iwork = Array(BlasInt, max(m,n))
    info = Array(BlasInt, 1)

    ccall((:dlatmr_, Base.liblapack_name), Void,
        (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{UInt8}, Ptr{BlasInt},
         Ptr{UInt8}, Ptr{Float64}, Ptr{BlasInt}, Ptr{Float64},
         Ptr{Float64}, Ptr{UInt8}, Ptr{UInt8}, Ptr{Float64},
         Ptr{BlasInt}, Ptr{Float64}, Ptr{Float64}, Ptr{BlasInt},
         Ptr{Float64}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt},
         Ptr{BlasInt}, Ptr{Float64}, Ptr{Float64}, Ptr{UInt8},
         Ptr{Float64}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        &m, &n, &dist, iseed,
        &sym, d, &mode, &cond,
        &dmax, &rsign, &grade, dl,
        &model, &condl, dr, &moder,
        &condr, &pivtng, ipivot, &kl,
        &ku, &sparse, &anorm, &pack,
        A, &lda, iwork, info)

    return A, info[1]
end

function latms!(m::BlasInt, n::BlasInt, dist::Char, iseed::Vector{BlasInt},
    sym::Char, d::Vector{Float64}, mode::BlasInt, cond::Float64, dmax::Float64,
    kl::BlasInt, ku::BlasInt, pack::Char)

    A = zeros(Float64, m, n)
    lda = m
    work = zeros(Float64, 3max(m, n))
    info = BlasInt[0]

    info = ccall((:LAPACKE_dlatms64_, Base.liblapack_name), BlasInt,
        (BlasInt, BlasInt, BlasInt, UInt8,
         Ptr{BlasInt}, UInt8, Ptr{Float64}, BlasInt,
         Float64, Float64, BlasInt, BlasInt,
         UInt8, Ptr{Float64}, BlasInt, Ptr{Float64}),
        LAPACK_COL_MAJOR, m, n, dist,
        iseed, sym, d, mode,
        cond, dmax, kl, ku,
        pack, A, lda, work)
    if info[1] != 0
        throw(LinAlg.LAPACKException(info[1]))
    end
    return d, A
end

function readLAPACKTestFile(fn::ByteString)
    ls = open(fn) do f
        readlines(f)
    end

    # Read test problem characters
    problem = symbol(shift!(ls)[1:3])

    # Read m
    if problem != :DGX && problem != :DXV
        nn = parse(Int, shift!(ls)[1:10])
    end

    # Read the values of m
    if problem == :SVD
        mval = map(t -> parse(Int, t), split(shift!(ls))[1:nn])
    end

    # Read values of p
    if problem in (:GLM, :GQR, :GSV, :CSD, :LSE)
        pval = map(t -> parse(Int, t), split(shift!(ls))[1:nn])
    else
        pval = Int[]
    end

    # Read the values of n
    if problem in (:SVD, :DBB, :GLM, :GQR, :GSV, :CSD, :LSE)
        nval = map(t -> parse(Int, t), split(shift!(ls))[1:nn])
    end

    # Read the number of values of K, followed by the values of K
    if problem in (:DSB, :DBB)
        error("not implemented yet")
    end

    if problem in (:DEV, :DES, :DVX, :DSX)
        error("not implemented yet")
    elseif problem in (:DGS, :DGX, :DGV, :DXV)
        error("not implemented yet")
    elseif !(problem in (:DSB, :GLM, :GQR, :GSV, :CSD, :LSE))

        # Read the number of parameter values.
        nparms = parse(Int, shift!(ls)[1:10])

        # Read the values of NB
        if problem != :DBB
            nbval = map(t -> parse(Int, t), split(shift!(ls))[1:nparms])
        end

        # Read the values of NBMIN
        if problem in (:NEP, :SEP, :SVD, :DGG)
            nbmin = map(t -> parse(Int, t), split(shift!(ls))[1:nparms])
        else
            nbmin = fill(1, nparms)
        end

        # Read the values of NX
        if problem in (:NEP, :SEP, :SVD)
            nxval = map(t -> parse(Int, t), split(shift!(ls))[1:nparms])
        else
            nxval = fill(1, nparms)
        end

        # Read the values of NSHIFT (if DGG) or NRHS (if SVD or DBB).
        if problem in (:SVD, :DBB, :DGG)
            nsval = map(t -> parse(Int, t), split(shift!(ls))[1:nparms])
        else
            nsval = fill(1, nparms)
        end

        # Read the values for MAXB.
        if problem == :DGG
            error("not implemented yet")
        else
            mxbval = fill(1, nparms)
        end
    end

    # Read the threshold value
    threshold = parse(Float64, shift!(ls)[1:10])

    if problem in (:SEP, :SVD, :DGG)

        # Read the flag that indicates whether to test LAPACK routines.
        tstchk = shift!(ls)[1]

        # Read the flag that indicates whether to test driver routines.
        tstdrv = shift!(ls)[1]
    end

    # Read the flag that indicates whether to test the error exits.

    tsterr = shift!(ls)[1]

    # Read the code describing how to set the random number seed.

    newsd = parse(Int, shift!(ls)[1])

    # if newsd = 2, read another line with 4 integers for the seed.

    if newsd == 2
        iseed = map(t -> parse(Int, t), split(shift!(ls))[1:10])
    else
        iseed = Int[0,0,0,1]
    end

    if !(problem in (:DGX, :DVX))
        tests = Dict{Symbol,Int}()
        for l in ls
            la = split(l)
            key = symbol(la[1])
            val = parse(Int, la[2])
            tests[key] = val
        end
    end

    return problem, Dict(:nn => nn, :mval => mval, :pval => pval, :nval => nval,
        :nparms => nparms, :nbval => nbval, :nbmin => nbmin, :nxval => nxval,
        :nsval => nsval, :mxbval => mxbval, :threshold => threshold,
        :tstchk => tstchk, :tstdrv => tstdrv, :tsterr => tsterr, :iseed => iseed,
        :tests => tests)
end

function latms!(m::BlasInt, n::BlasInt, dist::Char, iseed::Vector{BlasInt},
    sym::Char, d::Vector{Float64}, mode::BlasInt, cond::Float64, dmax::Float64,
    kl::BlasInt, ku::BlasInt, pack::Char)

    A = zeros(Float64, m, n)
    lda = m
    work = zeros(Float64, 3max(m, n))
    info = BlasInt[0]

    info = ccall((:LAPACKE_dlatms64_, Base.liblapack_name), BlasInt,
        (BlasInt, BlasInt, BlasInt, UInt8,
         Ptr{BlasInt}, UInt8, Ptr{Float64}, BlasInt,
         Float64, Float64, BlasInt, BlasInt,
         UInt8, Ptr{Float64}, BlasInt, Ptr{Float64}),
        LAPACK_COL_MAJOR, m, n, dist,
        iseed, sym, d, mode,
        cond, dmax, kl, ku,
        pack, A, lda, work)
    if info[1] != 0
        throw(LinAlg.LAPACKException(info[1]))
    end
    return d, A
end

function readLAPACKTestFile(fn::ByteString)
    ls = open(fn) do f
        readlines(f)
    end

    # Read test problem characters
    problem = symbol(shift!(ls)[1:3])

    # Read m
    if problem != :DGX && problem != :DXV
        nn = parse(Int, shift!(ls)[1:10])
    end

    # Read the values of m
    if problem == :SVD
        mval = map(t -> parse(Int, t), split(shift!(ls))[1:nn])
    end

    # Read values of p
    if problem in (:GLM, :GQR, :GSV, :CSD, :LSE)
        pval = map(t -> parse(Int, t), split(shift!(ls))[1:nn])
    else
        pval = Int[]
    end

    # Read the values of n
    if problem in (:SVD, :DBB, :GLM, :GQR, :GSV, :CSD, :LSE)
        nval = map(t -> parse(Int, t), split(shift!(ls))[1:nn])
    end

    # Read the number of values of K, followed by the values of K
    if problem in (:DSB, :DBB)
        error("not implemented yet")
    end

    if problem in (:DEV, :DES, :DVX, :DSX)
        error("not implemented yet")
    elseif problem in (:DGS, :DGX, :DGV, :DXV)
        error("not implemented yet")
    elseif !(problem in (:DSB, :GLM, :GQR, :GSV, :CSD, :LSE))

        # Read the number of parameter values.
        nparms = parse(Int, shift!(ls)[1:10])

        # Read the values of NB
        if problem != :DBB
            nbval = map(t -> parse(Int, t), split(shift!(ls))[1:nparms])
        end

        # Read the values of NBMIN
        if problem in (:NEP, :SEP, :SVD, :DGG)
            nbmin = map(t -> parse(Int, t), split(shift!(ls))[1:nparms])
        else
            nbmin = fill(1, nparms)
        end

        # Read the values of NX
        if problem in (:NEP, :SEP, :SVD)
            nxval = map(t -> parse(Int, t), split(shift!(ls))[1:nparms])
        else
            nxval = fill(1, nparms)
        end

        # Read the values of NSHIFT (if DGG) or NRHS (if SVD or DBB).
        if problem in (:SVD, :DBB, :DGG)
            nsval = map(t -> parse(Int, t), split(shift!(ls))[1:nparms])
        else
            nsval = fill(1, nparms)
        end

        # Read the values for MAXB.
        if problem == :DGG
            error("not implemented yet")
        else
            mxbval = fill(1, nparms)
        end
    end

    # Read the threshold value
    threshold = parse(Float64, shift!(ls)[1:10])

    if problem in (:SEP, :SVD, :DGG)

        # Read the flag that indicates whether to test LAPACK routines.
        tstchk = shift!(ls)[1]

        # Read the flag that indicates whether to test driver routines.
        tstdrv = shift!(ls)[1]
    end

    # Read the flag that indicates whether to test the error exits.

    tsterr = shift!(ls)[1]

    # Read the code describing how to set the random number seed.

    newsd = parse(Int, shift!(ls)[1])

    # if newsd = 2, read another line with 4 integers for the seed.

    if newsd == 2
        iseed = map(t -> parse(Int, t), split(shift!(ls))[1:10])
    else
        iseed = Int[0,0,0,1]
    end

    if !(problem in (:DGX, :DVX))
        tests = Dict{Symbol,Int}()
        for l in ls
            la = split(l)
            key = symbol(la[1])
            val = parse(Int, la[2])
            tests[key] = val
        end
    end

    return problem, Dict(:nn => nn, :mval => mval, :pval => pval, :nval => nval,
        :nparms => nparms, :nbval => nbval, :nbmin => nbmin, :nxval => nxval,
        :nsval => nsval, :mxbval => mxbval, :threshold => threshold,
        :tstchk => tstchk, :tstdrv => tstdrv, :tsterr => tsterr, :iseed => iseed,
        :tests => tests)
end

function testsvd{T}(::Type{T}, fn::ByteString)

    kmode  = [2*0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 0, 0]
    ktype  = [1, 2, 5*4, 5*6, 3*9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    maxtyp = 16
    ntypes = 16
    rtovfl = sqrt(realmax(T))
    rtunfl = sqrt(realmin(T))
    ulp    = eps(T)

    d = readLAPACKTestFile(fn)
    dd = d[2]
    nsizes = dd[:nn]
    iseed = dd[:iseed]

    for jsize = 1:nsizes
        m = dd[:mval][jsize]
        n = dd[:nval][jsize]
        mnmin = min(m, n)
        amninv = inv(max(m, n, 1))

        if nsizes != 1
            mtypes = min(maxtyp, ntypes)
        else
            mtypes = min(maxtyp + 1, ntypes)
        end

        for jtype = 1:mtypes
            if mtypes > maxtyp
                error("")
            end

            itype = ktype[jtype]
            imode = kmode[jtype]
            cond = inv(ulp)

            if jtype == 1
                anorm = 1.0
            elseif jtype == 2
                anorm = rtovfl*ulp*amninv
            elseif jtype == 3
                anorm = rtunfl*max(m, n)*inv(ulp)
            else
                anorm = one(T)
            end

            bidiag = false

            if itype == 1

                # zero matrix
                A = zeros(T, m, n)

            elseif itype == 2

                # identity matrix
                A = eye(T, m, n)*anorm

            elseif itype == 4

                # Diagonal Matrix, [Eigen]values Specified
                d, A = latms!(mnmin, mnmin, 'S', iseed, 'N', zeros(mnmin), imode, cond, anorm, 0, 0, 'N')
            end

            if itype in (1,2,4)
                display(svdvals(A))
                display(SVDModule.svdvals!(copy(A)))
            end
        end
    end
end

end # module
