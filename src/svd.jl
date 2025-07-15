using LinearAlgebra

import LinearAlgebra: mul!, rmul!

AdjointQtype = isdefined(LinearAlgebra, :AdjointQ) ? LinearAlgebra.AdjointQ : Adjoint

lmul!(G::LinearAlgebra.Givens, ::Nothing) = nothing
rmul!(::Nothing, G::LinearAlgebra.Givens) = nothing

function svdvals2x2(d1, d2, e)
    d1sq = d1 * d1
    d2sq = d2 * d2
    esq = e * e
    b = d1sq + d2sq + esq
    D = sqrt(abs2((d1 + d2) * (d1 - d2)) + esq * (2 * (d1sq + d2sq) + esq))
    D2 = b + D
    λ1 = 2 * d1sq * d2sq / D2
    λ2 = D2 / 2
    return minmax(sqrt(λ1), sqrt(λ2))
end

function svdIter!(
    B::Bidiagonal{T},
    n1,
    n2,
    shift,
    U = nothing,
    Vᴴ = nothing,
) where {T<:Real}

    if B.uplo === 'U'

        d = B.dv
        e = B.ev

        G, r = givens(d[n1] - abs2(shift) / d[n1], e[n1], n1, n1 + 1)
        lmul!(G, Vᴴ)

        ditmp = d[n1]
        ei1 = e[n1]
        di = ditmp * G.c + ei1 * G.s
        ei1 = -ditmp * G.s + ei1 * G.c
        di1 = d[n1+1]
        bulge = di1 * G.s
        di1 *= G.c

        for i = n1:n2-2
            G, r = givens(di, bulge, i, i + 1)
            rmul!(U, G')
            d[i] = G.c * di + G.s * bulge
            ei = G.c * ei1 + G.s * di1
            di1 = -G.s * ei1 + G.c * di1
            ei1 = e[i+1]
            bulge = G.s * ei1
            ei1 *= G.c

            G, r = givens(ei, bulge, i + 1, i + 2)
            lmul!(G, Vᴴ)
            e[i] = ei * G.c + bulge * G.s
            di = di1 * G.c + ei1 * G.s
            ei1 = -di1 * G.s + ei1 * G.c
            di2 = d[i+2]
            bulge = di2 * G.s
            di1 = di2 * G.c
        end

        G, r = givens(di, bulge, n2 - 1, n2)
        rmul!(U, G')
        d[n2-1] = G.c * di + G.s * bulge
        e[n2-1] = G.c * ei1 + G.s * di1
        d[n2] = -G.s * ei1 + G.c * di1

    else
        throw(ArgumentError("lower bidiagonal version not implemented yet"))
    end

    return B
end

# See LAWN 3
function svdDemmelKahan!(
    B::Bidiagonal{T},
    n1,
    n2,
    U = nothing,
    Vᴴ = nothing,
) where {T<:Real}

    if B.uplo === 'U'

        d = B.dv
        e = B.ev

        oldcs = one(T)
        G = LinearAlgebra.Givens(1, 2, one(T), zero(T))
        Gold = G

        for i = n1:n2-1
            G, r = givens(d[i] * G.c, e[i], i, i + 1)
            lmul!(G, Vᴴ)
            if i != n1
                e[i-1] = Gold.s * r
            end

            Gold, d[i] = givens(Gold.c * r, d[i+1] * G.s, i, i + 1)
            rmul!(U, Gold')
        end

        h = d[n2] * G.c
        e[n2-1] = h * Gold.s
        d[n2] = h * Gold.c

    else
        throw(ArgumentError("lower bidiagonal version not implemented yet"))
    end

    return B
end

# Recurrence to estimate smallest singular value from LAWN3 Lemma 1
function estimate_σ⁻!(
    dv::AbstractVector,
    ev::AbstractVector,
    n1::Integer,
    n2::Integer,
    tol::Real,
)

    # (4.3) p 18
    μ = abs(dv[n1])
    B1 = μ
    for j = n1:(n2-1)
        μ = abs(dv[j+1]) * (μ / (μ + abs(ev[j])))
        if abs(ev[j] / μ) < tol
            ev[j] = 0
        end
        B1 = min(B1, μ)
    end

    # (4.4) p 18
    λ = abs(dv[n2])
    B∞ = λ
    for j = (n2-1):-1:n1
        λ = abs(dv[j]) * (λ / (λ + abs(ev[j])))
        if abs(ev[j] / λ) < tol
            ev[j] = 0
        end
        B∞ = min(B∞, λ)
    end

    return min(B∞, B1)
end

# The actual SVD solver routine
# Notice that this routine doesn't adjust the sign and sorts the values
function __svd!(B::Bidiagonal{T}, args...; kwargs...) where {T<:Real}
    B.uplo == 'U' ? __svd_helper!(B, args...; kwargs...) : __svd_helper!(B', args...; kwargs...)
end

function __svd_helper!(
    B::Bidiagonal{T},
    U = nothing,
    Vᴴ = nothing;
    tol = 100eps(T),
) where {T<:Real}
    n = size(B, 1)
    if n == 0
        # Exit early in the empty case
        return nothing
    end
    n1 = 1
    n2 = n
    d = B.dv
    e = B.ev
    iteration = 0

    thresh = tol * zero(first(d))

    if B.uplo === 'U'
        while true
            @label top

            # Search for biggest index for non-zero off diagonal value in e
            for n2i = n2:-1:1
                if n2i == 1
                    # We are done!
                    return nothing
                else
                    if abs(e[n2i-1]) > thresh
                        n2 = n2i
                        break
                    end
                end
            end

            # Search for largest sub-bidiagonal matrix ending at n2
            for _n1 = (n2-1):-1:1
                if _n1 == 1
                    n1 = _n1
                    break
                elseif abs(e[_n1-1]) < thresh
                    n1 = _n1
                    break
                end
            end
            # @debug "Active submatrix" iteration n1 n2 d[n1] d[n2] e[n1] e[n2 - 1]

            # Deflation check. See LAWN3 p21
            # The Demmel-Kahan iteration moves the zero to the end and produces a
            # zero super diagonal element as well
            for i = n1:n2
                if d[i] == 0
                    # @debug "Deflation! Exact zero in diagonal element" i
                    svdDemmelKahan!(B, n1, n2, U, Vᴴ)

                    # We have now moved a zero to the end so the problem is one smaller
                    n2 -= 1

                    # We'll start over to find the relevant submatrix after deflation
                    @goto top
                end
            end

            # See LAWN 3 p 18 and
            # We approximate the smallest and largest singular values of the
            # current block to determine if it's safe to use shift or if
            # the zero shift algorithm is required to maintain high relative
            # accuracy
            #
            # We also set a new threshhold to be used in the next iteration
            σ⁻ = estimate_σ⁻!(d, e, n1, n2, tol)
            σ⁺ = max(maximum(view(d, n1:n2)), maximum(view(e, n1:(n2-1))))
            fudge = n2 - n1 + 1
            thresh = tol * σ⁻

            # @debug "estimated quantities" σ⁻ σ⁺ fudge thresh

            if fudge * tol * σ⁻ <= eps(σ⁺)
                svdDemmelKahan!(B, n1, n2, U, Vᴴ)
            else
                shift = svdvals2x2(d[n2-1], d[n2], e[n2-1])[1]
                if shift^2 < eps(B.dv[n1]^2)
                    # Shift is too small to affect the iteration so just use
                    # zero-shift algorithm anyway
                    svdDemmelKahan!(B, n1, n2, U, Vᴴ)
                else
                    svdIter!(B, n1, n2, shift, U, Vᴴ)
                end
            end

            iteration += 1
        end
    else
        # Just transpose the matrix
        error("SVD for lower triangular bidiagonal matrices isn't implemented yet.")
    end
end

function _svdvals!(B::Bidiagonal{T}; tol = eps(T)) where {T<:Real}
    __svd!(B, tol = tol)
    return sort(abs.(diag(B)), rev = true)
end

# We want to avoid that zero singular values zero out any singular vectors so we'd
# have to use a modified sign function
_sign(x) = ifelse(iszero(x), one(x), sign(x))

function _sort_and_adjust!(U, s, Vᴴ)
    n = length(s)

    # Adjust sign of singular values if necessary
    if any(!isposdef, s)
        rmul!(U, Diagonal(_sign.(s)))
        map!(abs, s, s)
    end

    # Sort singular values if necessary
    # FixMe! Try to do this without (much) allocation
    p = sortperm(s, by = abs, rev = true)
    if p != 1:n
        U[:] = U[:, p]
        s[:] = s[p]
        Vᴴ[:] = Vᴴ[p, :]
    end

    return nothing
end

function _svd!(B::Bidiagonal{T}; tol = eps(T)) where {T<:Real}
    n = size(B, 1)

    U = Matrix{T}(I, n, n)
    Vᴴ = Matrix{T}(I, n, n)

    __svd!(B, U, Vᴴ, tol = tol)

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
            LinearAlgebra.reflectorApply!(x, τi, view(A, i:m, (i+1):n))
            if i < n
                x = view(A, i, i+1:n)
                conj!(x)
                τi = LinearAlgebra.reflector!(x)
                push!(τr, τi)
                LinearAlgebra.reflectorApply!(view(A, (i+1):m, (i+1):n), x, τi)
            end
        end

        bd = Bidiagonal(real(diag(A)), real(diag(A, 1)), :U)

        return BidiagonalFactorization{eltype(bd),typeof(bd.dv),typeof(A),typeof(τl)}(
            bd,
            A,
            τl,
            τr,
        )
    else
        # wide case: lower bidiagonal
        for i = 1:min(m, n)
            x = view(A, i, i:n)
            conj!(x)
            τi = LinearAlgebra.reflector!(x)
            push!(τr, τi)
            LinearAlgebra.reflectorApply!(view(A, (i+1):m, i:n), x, τi)
            if i < m
                x = view(A, i+1:m, i)
                τi = LinearAlgebra.reflector!(x)
                push!(τl, τi)
                LinearAlgebra.reflectorApply!(x, τi, view(A, (i+1):m, (i+1):n))
            end
        end

        bd = Bidiagonal(real(diag(A)), real(diag(A, -1)), :L)

        return BidiagonalFactorization{eltype(bd),typeof(bd.dv),typeof(A),typeof(τl)}(
            bd,
            A,
            τl,
            τr,
        )
    end
end

# This one is currently very type unstable but it is useful to be able to reuse
# existing elementary reflectors types such as QRPackedQ, LQPackedQ, and HessenbergQ
# such that multiplication methods are available for free.
function Base.getproperty(F::BidiagonalFactorization, s::Symbol)
    BD = getfield(F, :bidiagonal)
    R = getfield(F, :reflectors)
    τl = getfield(F, :τl)
    τr = getfield(F, :τr)
    if BD.uplo === 'U'
        if s === :leftQ
            return LinearAlgebra.QRPackedQ(R, τl)
        elseif s === :rightQ
            factors = copy(transpose(R[1:size(R, 2), :]))
            return LinearAlgebra.HessenbergQ{
                eltype(factors),
                typeof(factors),
                typeof(τr),
                false,
            }(
                'U',
                factors,
                τr,
            )
        else
            return getfield(F, s)
        end
    else
        if s === :leftQ
            factors = R[:, 1:size(R, 1)]
            return LinearAlgebra.HessenbergQ{
                eltype(factors),
                typeof(factors),
                typeof(τr),
                false,
            }(
                'U',
                factors,
                τl,
            )
        elseif s === :rightQ
            # FIXME! Can I get the LQPackedQ versino going to avoid the copy?
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

    for j = 1:n
        for l = length(Q.τ):-1:1
            τl = Q.τ[l]
            ṽ = view(Q.factors, (l+2):m, l)
            b = view(B, (l+2):m, j)
            vᴴb = B[l+1, j]
            if length(ṽ) > 0
                vᴴb += ṽ'b
            end
            B[l+1, j] -= τl * vᴴb
            b .-= ṽ .* τl .* vᴴb
        end
    end

    return B
end

function lmul!(adjQ::AdjointQtype{<:Any,<:LinearAlgebra.HessenbergQ}, B::AbstractVecOrMat)

    Q = parent(adjQ)
    m, n = size(B, 1), size(B, 2)

    if m != size(adjQ, 2)
        throw(DimensionMismatch(""))
    end

    for j = 1:n
        for l = 1:length(Q.τ)
            τl = Q.τ[l]
            ṽ = view(Q.factors, (l+2):m, l)
            b = view(B, (l+2):m, j)
            vᴴb = B[l+1, j]
            if length(ṽ) > 0
                vᴴb += ṽ'b
            end
            B[l+1, j] -= τl' * vᴴb
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

    for i = 1:m
        for l = 1:(n-1)
            τl = Q.τ[l]
            ṽ = view(Q.factors, (l+2):n, l)
            aᵀ = transpose(view(A, i, (l+2):n))
            aᵀv = A[i, l+1]
            if length(ṽ) > 0
                aᵀv += aᵀ * ṽ
            end
            A[i, l+1] -= aᵀv * τl
            aᵀ .-= aᵀv .* τl .* ṽ'
        end
    end

    return A
end

function rmul!(A::AbstractMatrix, adjQ::AdjointQtype{<:Any,<:LinearAlgebra.HessenbergQ})
    m, n = size(A)
    Q = parent(adjQ)

    if n != size(adjQ, 1)
        throw(DimensionMismatch(""))
    end

    for i = 1:m
        for l = (n-1):-1:1
            τl = Q.τ[l]
            ṽ = view(Q.factors, (l+2):n, l)
            aᵀ = transpose(view(A, i, (l+2):n))
            aᵀv = A[i, l+1]
            if length(ṽ) > 0
                aᵀv += aᵀ * ṽ
            end
            A[i, l+1] -= aᵀv * τl'
            aᵀ .-= aᵀv .* τl' .* ṽ'
        end
    end

    return A
end

# FIXME! Commented out for now because we currently don't produce
# any LQPackedQ matrices.
# function rmul!(A::AbstractMatrix, Q::LinearAlgebra.LQPackedQ)

#     m, n = size(A)

#     if n != size(Q, 1)
#         throw(DimensionMismatch(""))
#     end

#     for i = 1:m
#         for l = length(Q.τ):-1:1
#             τl = Q.τ[l]
#             ṽ = view(Q.factors, l, (l+1):n)
#             aᵀ = transpose(view(A, i, (l+1):n))
#             aᵀv = A[i, l]
#             if length(ṽ) > 0
#                 aᵀv += aᵀ * ṽ
#             end
#             A[i, l] -= aᵀv * τl
#             aᵀ .-= aᵀv .* τl .* ṽ'
#         end
#     end

#     return A
# end

# Overload LinearAlgebra methods

LinearAlgebra.svdvals!(B::Bidiagonal{T}; tol = eps(T)) where {T<:Real} =
    _svdvals!(B, tol = tol)

"""
    svdvals!(A [, tol])

Generic computation of singular values.
```jldoctest
julia> using LinearAlgebra, GenericLinearAlgebra, Quaternions

julia> n = 20;

julia> H = [big(1)/(i + j - 1) for i in 1:n, j in 1:n]; # The Hilbert matrix

julia> Float64(svdvals(H)[end]/svdvals(Float64.(H))[end] - 1) # The relative error of the LAPACK based solution in 64 bit floating point.
-0.9999999999447275

julia> A = qr([Quaternion(randn(4)...) for i in 1:3, j in 1:3]).Q *
           Diagonal([3, 2, 1]) *
           qr([Quaternion(randn(4)...) for i in 1:3, j in 1:3]).Q'; # A quaternion matrix with the singular value 1, 2, and 3.

julia> svdvals(A) ≈ [3, 2, 1]
true
```
"""
function LinearAlgebra.svdvals!(A::StridedMatrix; tol = eps(real(eltype(A))))
    B = bidiagonalize!(A).bidiagonal
    # It doesn't matter that we transpose the bidiagonal matrix when we are only interested in the values
    return svdvals!(Bidiagonal(B.dv, B.ev, :U), tol = tol)
end

# FixMe! The full keyword is redundant for Bidiagonal and should be removed from Base
LinearAlgebra.svd!(
    B::Bidiagonal{T};
    tol = eps(T),
    full = false,
    # To avoid breaking on <Julia 1.3, the `alg` keyword doesn't do anything. Once we drop support for Julia 1.2
    # and below, we can make the keyword argument work correctly
    alg = nothing,
) where {T<:Real} = _svd!(B, tol = tol)

"""
    svd!(A[, tol, full])::SVD

A generic singular value decomposition (SVD). The implementation only uses Julia functions so the SVD can be computed for any element type provided that the necessary arithmetic operations are supported by the element type.

- `tol`: The relative tolerance for determining convergence. The default value is `eltype(T)` where `T` is the element type of the input matrix bidiagonal (i.e. after converting the matrix to bidiagonal form).

- `full`: If set to `true` then all the left and right singular vectors are returned. If set to `false` then only the vectors corresponding to the number of rows and columns of the input matrix `A` are returned (the default).

# Algorithm
...tomorrow

# Example

```jldoctest
julia> svd(big.([1 2; 3 4]))
SVD{BigFloat, BigFloat, Matrix{BigFloat}}
U factor:
2×2 Matrix{BigFloat}:
 -0.404554   0.914514
 -0.914514  -0.404554
singular values:
2-element Vector{BigFloat}:
 5.464985704219042650451188493284182533042584640492784181017488774646871847029449
 0.3659661906262578204229643842614005434788136943931877734325179702209382149672422
Vt factor:
2×2 Matrix{BigFloat}:
 -0.576048  -0.817416
 -0.817416   0.576048
```
"""
function LinearAlgebra.svd!(
    A::StridedMatrix{T};
    tol = eps(real(eltype(A))),
    full = false,
    # To avoid breaking on <Julia 1.3, the `alg` keyword doesn't do anything. Once we drop support for Julia 1.2
    # and below, we can make the keyword argument work correctly
    alg = nothing,
) where {T}

    m, n = size(A)

    # Convert the input matrix A to bidiagonal form and return a BidiagonalFactorization object
    BF = bidiagonalize!(A)

    # The location of the super/sub-diagonal of the bidiagonal matrix depends on the aspect ratio of the input. For tall and square matrices, the bidiagonal matrix is upper whereas it is lower for wide input matrices as illustrated below. The 'O' entries indicate orthogonal (unitary) matrices.

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
    B = _B.uplo === 'U' ? _B : Bidiagonal(_B.dv, _B.ev, :U)

    # Compute the SVD of the bidiagonal matrix B
    F = _svd!(B, tol = tol)

    # Form the matrices U and Vᴴ by combining the singular vector matrices of the bidiagonal SVD with the Householder reflectors from the bidiagonal factorization.
    if _B.uplo === 'U'
        U = Matrix{T}(I, m, full ? m : n)
        U[1:n, 1:n] = F.U
        lmul!(BF.leftQ, U)
        Vᴴ = F.Vt * BF.rightQ'
    else
        U = BF.leftQ * F.V
        Vᴴ = Matrix{T}(I, full ? n : m, n)
        Vᴴ[1:m, 1:m] = F.U'
        rmul!(Vᴴ, BF.rightQ')
    end

    s = F.S

    return SVD(U, s, Vᴴ)
end
