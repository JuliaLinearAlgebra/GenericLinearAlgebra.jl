using LinearAlgebra

import LinearAlgebra: mul!, rmul!

lmul!(G::LinearAlgebra.Givens, ::Nothing) = nothing
rmul!(::Nothing, G::LinearAlgebra.Givens) = nothing

function svdvals2x2(d1, d2, e)
    d1sq = d1*d1
    d2sq = d2*d2
    esq  = e*e
    b = d1sq + d2sq + esq
    D = sqrt(abs2((d1 + d2)*(d1 - d2)) + esq*(2*(d1sq + d2sq) + esq))
    D2 = b + D
    λ1 = 2*d1sq*d2sq/D2
    λ2 = D2/2
    return minmax(sqrt(λ1), sqrt(λ2))
end

function svdIter!(B::Bidiagonal{T}, n1, n2, shift, U = nothing, Vᴴ = nothing) where T<:Real

    if istriu(B)

        d = B.dv
        e = B.ev

        G, r = givens(d[n1] - abs2(shift)/d[n1], e[n1], 1, 2)
        lmul!(G, Vᴴ)

        ditmp       = d[n1]
        ei1         = e[n1]
        di          = ditmp*G.c + ei1*G.s
        ei1         = -ditmp*G.s + ei1*G.c
        di1         = d[n1 + 1]
        bulge       = di1*G.s
        di1        *= G.c

        for i = n1:n2 - 2
            G, r      = givens(di, bulge, i, i + 1)
            rmul!(U, G')
            d[i]      = G.c*di + G.s*bulge
            ei        = G.c*ei1 + G.s*di1
            di1       = -G.s*ei1 + G.c*di1
            ei1       = e[i + 1]
            bulge     = G.s*ei1
            ei1      *= G.c

            G, r      = givens(ei, bulge, i + 1, i + 2)
            lmul!(G, Vᴴ)
            e[i]      = ei*G.c + bulge*G.s
            di        = di1*G.c + ei1*G.s
            ei1       = -di1*G.s + ei1*G.c
            di2       = d[i + 2]
            bulge     = di2*G.s
            di1       = di2*G.c
        end

        G, r      = givens(di, bulge, n2 - 1, n2)
        rmul!(U, G')
        d[n2 - 1] = G.c*di + G.s*bulge
        e[n2 - 1] = G.c*ei1 + G.s*di1
        d[n2]     = -G.s*ei1 + G.c*di1

    else
        throw(ArgumentError("lower bidiagonal version not implemented yet"))
    end

    return B
end

# See LAWN 3
function svdDemmelKahan!(B::Bidiagonal{T}, n1, n2, U = nothing, Vᴴ = nothing) where T<:Real

    if istriu(B)

        d = B.dv
        e = B.ev

        oldcs = one(T)
        G     = LinearAlgebra.Givens(1, 2, one(T), zero(T))
        Gold  = G

        for i = n1:n2 - 1
            G, r  = givens(d[i] * G.c, e[i], i, i + 1)
            lmul!(G, Vᴴ)
            if i != n1
                e[i - 1] = Gold.s * r
            end

            Gold, d[i] = givens(Gold.c * r, d[i + 1] * G.s, i, i + 1)
            rmul!(U, Gold')
        end

        h         = d[n2] * G.c
        e[n2 - 1] = h * Gold.s
        d[n2]     = h * Gold.c

    else
        throw(ArgumentError("lower bidiagonal version not implemented yet"))
    end

    return B
end

# The actual SVD solver routine
# Notice that this routine doesn't adjust the sign and sorts the values
function __svd!(B::Bidiagonal{T}, U = nothing, Vᴴ = nothing; tol = eps(T), debug = false) where T<:Real

    n = size(B, 1)
    n1 = 1
    n2 = n
    d = B.dv
    e = B.ev
    count = 0

    if istriu(B)
        while true

            # Search for biggest index for non-zero off diagonal value in e
            for n2i = n2:-1:1
                if n2i == 1
                    # We are done!

                    return nothing
                else
                    tolcritTop = tol * abs(d[n2i - 1] * d[n2i])

                    # debug && println("n2i=", n2i, ", d[n2i-1]=", d[n2i-1], ", d[n2i]=", d[n2i],
                        # ", e[n2i-1]=", e[n2i-1], ", tolcritTop=", tolcritTop)

                    if abs(e[n2i - 1]) > tolcritTop
                        n2 = n2i
                        break
                    end
                end
            end

            # Search for largest sub-bidiagonal matrix ending at n2
            for _n1 = (n2 - 1):-1:1

                if _n1 == 1
                    n1 = _n1
                    break
                else
                    tolcritBottom = tol * abs(d[_n1 - 1] * d[_n1])

                    # debug && println("n1=", n1, ", d[n1]=", d[n1], ", d[n1-1]=", d[n1-1], ", e[n1-1]", e[n1-1],
                        # ", tolcritBottom=", tolcritBottom)

                    if abs(e[_n1 - 1]) < tolcritBottom
                        n1 = _n1
                        break
                    end
                end
            end

            debug && println("n1=", n1, ", n2=", n2, ", d[n1]=", d[n1], ", d[n2]=", d[n2], ", e[n1]=", e[n1])

            # FixMe! Calling the zero shifted version only the first time is adhoc but seems
            # to work well. Reexamine analysis in LAWN 3 to improve this later.
            if count == 0
                svdDemmelKahan!(B, n1, n2, U, Vᴴ)
            else
                shift = svdvals2x2(d[n2 - 1], d[n2], e[n2 - 1])[1]
                svdIter!(B, n1, n2, shift, U, Vᴴ)
            end
            count += 1
            debug && println("count=", count)
        end
    else
        # Just transpose the matrix
        error("")
        # return _svd!(Bidiagonal(d, e, :U), Vᴴ, U; tol = tol, debug = debug)
    end
end

function _svdvals!(B::Bidiagonal{T}; tol = eps(T), debug = false) where T<:Real
    __svd!(B, tol = tol, debug = debug)
    return sort(abs.(diag(B)), rev = true)
end

function _sort_and_adjust!(U, s, Vᴴ)
    n = length(s)

    # Adjust sign of singular values if necessary
    if any(!isposdef, s)
        rmul!(U, Diagonal(sign.(s)))
        map!(abs, s, s)
    end

    # Sort singular values if necessary
    # FixMe! Try to do this without (much) allocation
    p = sortperm(s, by = abs, rev = true)
    if p != 1:n
        U[:]  = U[:,p]
        s[:]  = s[p]
        Vᴴ[:] = Vᴴ[p,:]
    end

    return nothing
end

function _svd!(B::Bidiagonal{T}; tol = eps(T), debug = false) where T<:Real
    n  = size(B, 1)

    U  = Matrix{T}(I, n, n)
    Vᴴ = Matrix{T}(I, n, n)

    __svd!(B, U, Vᴴ, tol = tol, debug = debug)

    s = diag(B)

    # Sort and adjust the singular values. The algorithm used in __svd! might produce negative singular values and it will return the values in no particular ordering.
    _sort_and_adjust!(U, s, Vᴴ)

    return SVD(U, s, Vᴴ)
end

struct BidiagonalFactorization{T,S,U,V}
    bidiagonal::Bidiagonal{T,S}
    reflectors::U
    τl::V
    τr::V
end

function bidiagonalize!(A::AbstractMatrix)
    m, n = size(A)
    τl, τr = eltype(A)[], eltype(A)[]

    if m >= n
        # tall case: upper bidiagonal
        for i = 1:min(m, n)
            x = view(A, i:m, i)
            τi = LinearAlgebra.reflector!(x)
            push!(τl, τi)
            LinearAlgebra.reflectorApply!(x, τi, view(A, i:m, (i + 1):n))
            if i < n
                x = view(A, i, i + 1:n)
                conj!(x)
                τi = LinearAlgebra.reflector!(x)
                push!(τr, τi)
                LinearAlgebra.reflectorApply!(view(A, (i + 1):m, (i + 1):n), x, τi)
            end
        end

        bd = Bidiagonal(real(diag(A)), real(diag(A, 1)), :U)

        return BidiagonalFactorization{eltype(bd),typeof(bd.dv),typeof(A),typeof(τl)}(bd, A, τl, τr)
    else
        # wide case: lower bidiagonal
        for i = 1:min(m, n)
            x = view(A, i, i:n)
            conj!(x)
            τi = LinearAlgebra.reflector!(x)
            push!(τr, τi)
            LinearAlgebra.reflectorApply!(view(A, (i + 1):m, i:n), x, τi)
            if i < m
                x = view(A, i + 1:m, i)
                τi = LinearAlgebra.reflector!(x)
                push!(τl, τi)
                LinearAlgebra.reflectorApply!(x, τi, view(A, (i + 1):m, (i + 1):n))
            end
        end

        bd = Bidiagonal(real(diag(A)), real(diag(A, -1)), :L)

        return BidiagonalFactorization{eltype(bd),typeof(bd.dv),typeof(A),typeof(τl)}(bd, A, τl, τr)
    end
end

# This one is currently very type unstable but it is useful to be able to reuse
# existing elementary reflectors types such as QRPackedQ, LQPackedQ, and HessenbergQ
# such that multiplication methods are available for free.
function Base.getproperty(F::BidiagonalFactorization, s::Symbol)
    BD = getfield(F, :bidiagonal)
    R  = getfield(F, :reflectors)
    τl = getfield(F, :τl)
    τr = getfield(F, :τr)
    if istriu(BD)
        if s === :leftQ
            return LinearAlgebra.QRPackedQ(R, τl)
        elseif s === :rightQ
            return LinearAlgebra.HessenbergQ(copy(transpose(R[1:size(R,2),:])), τr)
        else
            return getfield(F, s)
        end
    else
        if s === :leftQ
            return LinearAlgebra.HessenbergQ(R[:,1:size(R,1)], τl)
        elseif s === :rightQ
            # return transpose(LinearAlgebra.LQPackedQ(R, τr)) # FixMe! check that this shouldn't be adjoint
            LinearAlgebra.QRPackedQ(copy(transpose(R)), τr)
        else
            return getfield(F, s)
        end
    end
end

# For now, we define a generic lmul! and rmul! for HessenbergQ here
function lmul!(Q::LinearAlgebra.HessenbergQ, B::AbstractVecOrMat)

    m, n = size(B, 1), size(B, 2)

    if m != size(Q, 2)
        throw(DimensionMismatch(""))
    end

    for j in 1:n
        for l in length(Q.τ):-1:1
            τl = Q.τ[l]
            ṽ  = view(Q.factors, (l + 2):m, l)
            b  = view(B, (l + 2):m, j)
            vᴴb = B[l + 1, j]
            if length(ṽ) > 0
                vᴴb += ṽ'b
            end
            B[l + 1, j] -= τl*vᴴb
            b .-= ṽ .* τl .* vᴴb
        end
    end

    return B
end

function lmul!(adjQ::Adjoint{<:Any,<:LinearAlgebra.HessenbergQ}, B::AbstractVecOrMat)

    Q = parent(adjQ)
    m, n = size(B, 1), size(B, 2)

    if m != size(adjQ, 2)
        throw(DimensionMismatch(""))
    end

    for j in 1:n
        for l in 1:length(Q.τ)
            τl = Q.τ[l]
            ṽ  = view(Q.factors, (l + 2):m, l)
            b  = view(B, (l + 2):m, j)
            vᴴb = B[l + 1, j]
            if length(ṽ) > 0
                vᴴb += ṽ'b
            end
            B[l + 1, j] -= τl'*vᴴb
            b .-= ṽ .* τl' .* vᴴb
        end
    end

    return B
end

function rmul!(A::AbstractMatrix, Q::LinearAlgebra.HessenbergQ)

    m, n = size(A)

    if n != size(Q, 1)
        throw(DimensionMismatch(""))
    end

    for i in 1:m
        for l in 1:(n - 1)
            τl = Q.τ[l]
            ṽ  = view(Q.factors, (l + 2):n, l)
            aᵀ = transpose(view(A, i, (l + 2):n))
            aᵀv = A[i, l + 1]
            if length(ṽ) > 0
                aᵀv += aᵀ*ṽ
            end
            A[i, l + 1] -= aᵀv*τl
            aᵀ .-= aᵀv .* τl .* ṽ'
        end
    end

    return A
end

function rmul!(A::AbstractMatrix, adjQ::Adjoint{<:Any,<:LinearAlgebra.HessenbergQ})

    m, n = size(A)
    Q = parent(adjQ)

    if n != size(adjQ, 1)
        throw(DimensionMismatch(""))
    end

    for i in 1:m
        for l in (n - 1):-1:1
            τl = Q.τ[l]
            ṽ  = view(Q.factors, (l + 2):n, l)
            aᵀ = transpose(view(A, i, (l + 2):n))
            aᵀv = A[i, l + 1]
            if length(ṽ) > 0
                aᵀv += aᵀ*ṽ
            end
            A[i, l + 1] -= aᵀv*τl'
            aᵀ .-= aᵀv .* τl' .* ṽ'
        end
    end

    return A
end

function rmul!(A::AbstractMatrix, Q::LinearAlgebra.LQPackedQ)

    m, n = size(A)

    if n != size(Q, 1)
        throw(DimensionMismatch(""))
    end

    for i in 1:m
        for l in length(Q.τ):-1:1
            τl = Q.τ[l]
            ṽ  = view(Q.factors, l, (l + 1):n)
            aᵀ = transpose(view(A, i, (l + 1):n))
            aᵀv = A[i, l]
            if length(ṽ) > 0
                aᵀv += aᵀ*ṽ
            end
            A[i, l] -= aᵀv*τl
            aᵀ .-= aᵀv .* τl .* ṽ'
        end
    end

    return A
end

# Overload LinearAlgebra methods
LinearAlgebra.svdvals!(B::Bidiagonal{T}; tol = eps(T), debug = false) where T<:Real = _svdvals!(B, tol = tol, debug = debug)
function LinearAlgebra.svdvals!(A::StridedMatrix; tol = eps(real(eltype(A))), debug = false)
    B = bidiagonalize!(A).bidiagonal
    # It doesn't matter that we transpose the bidiagonal matrix when we are only interested in the values
    return svdvals!(Bidiagonal(B.dv, B.ev, :U), tol = tol, debug = debug)
end

# FixMe! The full keyword is redundant for Bidiagonal and should be removed from Base
LinearAlgebra.svd!(B::Bidiagonal{T}; tol = eps(T), full = false, debug = false) where T<:Real = _svd!(B, tol = tol, debug = debug)

"""
    svd!

A generic SVD implementation.
"""
function LinearAlgebra.svd!(A::StridedMatrix{T}; tol = eps(real(eltype(A))), full = false, debug = false) where T

    m, n = size(A)

    # Convert the input matrix A to bidiagonal form and return a BidiagonalFactorization object
    BF = bidiagonalize!(A)

    # The location of the super/sub-diagonal of the bidiagonal matrix depends on the aspect ratio of the input. For tall and square matrices, the bidiagonal matrix is upper whereas it is lower for wide input matrices as illustrated below. The 'O' entried indicate orthogonal (unitary) matrices.

    #      A      =   Q_l  *   B   * Q_r^H

    #     |x x x| = |O O O| |x x  | |O O O|
    #     |x x x|   |O O O|*|  x x|*|O O O|
    #     |x x x|   |O O O| |    x| |O O O|
    #     |x x x|   |O O O|
    #     |x x x|   |O O O|

    # |x x x x x| = |O O O| |x    | |O O O O O|
    # |x x x x x|   |O O O|*|x x  |*|O O O O O|
    # |x x x x x|   |O O O| |  x x| |O O O O O|

    _B = BF.bidiagonal
    B  = istriu(_B) ? _B : Bidiagonal(_B.dv, _B.ev, :U)

    # Compute the SVD of the bidiagonal matrix B
    F = _svd!(B, tol = tol, debug = debug)

    # Form the matrices U and Vᴴ by combining the singular vector matrices of the bidiagonal SVD with the Householder reflectors from the bidiagonal factorization.
    if istriu(_B)
        U  = BF.leftQ*F.U
        Vᴴ = F.Vt*BF.rightQ'
    else
        U  = BF.leftQ*F.V
        Vᴴ = (BF.rightQ*F.U)'
    end

    s = F.S

    return SVD(U, s, Vᴴ)
end
