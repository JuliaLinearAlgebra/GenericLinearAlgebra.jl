using Printf
using LinearAlgebra
using LinearAlgebra: Givens, Rotation

# Auxiliary
function adiagmax(A::StridedMatrix)
    adm = zero(typeof(real(A[1])))
    @inbounds begin
        for i in size(A, 1)
            adm = max(adm, abs(A[i, i]))
        end
    end
    return adm
end

# Hessenberg Matrix
struct HessenbergMatrix{T,S<:StridedMatrix} <: AbstractMatrix{T}
    data::S
end

Base.copy(H::HessenbergMatrix{T,S}) where {T,S} = HessenbergMatrix{T,S}(copy(H.data))

Base.getindex(H::HessenbergMatrix{T,S}, i::Integer, j::Integer) where {T,S} =
    i > j + 1 ? zero(T) : H.data[i, j]

Base.size(H::HessenbergMatrix) = size(H.data)
Base.size(H::HessenbergMatrix, i::Integer) = size(H.data, i)

function LinearAlgebra.ldiv!(H::HessenbergMatrix, B::AbstractVecOrMat)
    n = size(H, 1)
    Hd = H.data
    for i = 1:n-1
        G, _ = givens!(Hd, i, i + 1, i)
        lmul!(G, view(Hd, 1:n, i+1:n))
        lmul!(G, B)
    end
    ldiv!(Triangular(Hd, :U), B)
end

# Hessenberg factorization
struct HessenbergFactorization{T,S<:StridedMatrix,U} <: Factorization{T}
    data::S
    τ::Vector{U}
end

function _hessenberg!(A::StridedMatrix{T}) where {T}
    n = LinearAlgebra.checksquare(A)
    τ = Vector{Householder{T}}(undef, n - 1)
    for i = 1:n-1
        xi = view(A, i+1:n, i)
        t = LinearAlgebra.reflector!(xi)
        H = Householder{T,typeof(xi)}(view(xi, 2:n-i), t)
        τ[i] = H
        lmul!(H', view(A, i+1:n, i+1:n))
        rmul!(view(A, :, i+1:n), H)
    end
    return HessenbergFactorization{T,typeof(A),eltype(τ)}(A, τ)
end
LinearAlgebra.hessenberg!(A::StridedMatrix) = _hessenberg!(A)

Base.size(H::HessenbergFactorization, args...) = size(H.data, args...)

# Schur
struct Schur{T,S<:StridedMatrix} <: Factorization{T}
    data::S
    R::Rotation{T}
end

function Base.getproperty(F::Schur, s::Symbol)
    if s === :T || s === :Schur
        return triu(F.data, -1)
    else
        return getfield(F, s)
    end
end

function wilkinson(Hmm, t, d)
    λ1 = (t + sqrt(t * t - 4d)) / 2
    λ2 = (t - sqrt(t * t - 4d)) / 2
    return ifelse(abs(Hmm - λ1) < abs(Hmm - λ2), λ1, λ2)
end

# We currently absorb extra unsupported keywords in kwargs. These could e.g. be scale and permute. Do we want to check that these are false?
function _schur!(
    H::HessenbergFactorization{T};
    tol = eps(real(T)),
    shiftmethod = :Francis,
    maxiter = 30 * size(H, 1),
    kwargs...,
) where {T}

    n = size(H, 1)
    istart = 1
    iend = n
    HH = H.data
    τ = Rotation(Givens{T}[])

    # iteration count
    i = 0

    @inbounds while true
        i += 1
        @debug "iteration" i
        if i > maxiter
            throw(ArgumentError("iteration limit $maxiter reached"))
        end

        # Determine if the matrix splits. Find lowest positioned subdiagonal "zero"
        for _istart = iend-1:-1:1
            if abs(HH[_istart+1, _istart]) <=
               tol * (abs(HH[_istart, _istart]) + abs(HH[_istart+1, _istart+1]))
                # Check if subdiagonal element H[i+1,i] is "zero" such that we can split the matrix

                # Set istart to the beginning of the second block
                istart = _istart + 1

                @debug "Split!" HH[istart, istart-1] istart

                # Set the subdiagonal element to zero to signal that a split has taken place
                HH[istart, istart-1] = 0
                break
            end

            # If no splits have taken place then we start from the top
            istart = 1
        end

        # if block size is one we deflate
        if istart >= iend
            @debug "Bottom deflation! Block size is one. New iend is" iend - 1
            iend -= 1

            # and the same for a 2x2 block
        elseif istart + 1 == iend
            @debug "Bottom deflation! Block size is two. New iend is" iend - 2
            iend -= 2

            # run a QR iteration
            # shift method is specified with shiftmethod kw argument
        else
            Hmm = HH[iend, iend]
            Hm1m1 = HH[iend-1, iend-1]
            if iszero(i % 10)
                # Use Eispack exceptional shift
                β = abs(HH[iend, iend-1]) + abs(HH[iend-1, iend-2])
                d = (Hmm + β)^2 - Hmm * β / 2
                t = 2 * Hmm + 3 * β / 2
            else
                d = Hm1m1 * Hmm - HH[iend, iend-1] * HH[iend-1, iend]
                t = Hm1m1 + Hmm
            end
            @debug "block start is, block end, d, and t" istart iend d t

            if shiftmethod == :Francis
                @debug "Francis double shift!" HH[iend, iend-1] HH[iend-1, iend-2]

                doubleShiftQR!(HH, τ, t, d, istart, iend)
            elseif shiftmethod == :Rayleigh
                @debug "Single shift with Rayleigh shift!" HH[iend, iend-1]

                # Run a bulge chase
                singleShiftQR!(HH, τ, Hmm, istart, iend)
            else
                throw(
                    ArgumentError(
                        "only support supported shift methods are :Francis (default) and :Rayleigh. You supplied $shiftmethod",
                    ),
                )
            end
        end
        if iend <= 2
            break
        end
    end

    return Schur{T,typeof(HH)}(HH, τ)
end
_schur!(A::StridedMatrix; kwargs...) = _schur!(_hessenberg!(A); kwargs...)
LinearAlgebra.schur!(A::StridedMatrix; kwargs...) = _schur!(A; kwargs...)

function singleShiftQR!(
    HH::StridedMatrix,
    τ::Rotation,
    shift::Number,
    istart::Integer,
    iend::Integer,
)
    m = size(HH, 1)
    H11 = HH[istart, istart]
    H21 = HH[istart+1, istart]
    if m > istart + 1
        Htmp = HH[istart+2, istart]
        HH[istart+2, istart] = 0
    end
    G, _ = givens(H11 - shift, H21, istart, istart + 1)
    lmul!(G, view(HH, :, istart:m))
    rmul!(view(HH, 1:min(istart + 2, iend), :), G')
    lmul!(G, τ)
    for i = istart:iend-2
        G, _ = givens(HH[i+1, i], HH[i+2, i], i + 1, i + 2)
        lmul!(G, view(HH, :, i:m))
        HH[i+2, i] = Htmp
        if i < iend - 2
            Htmp = HH[i+3, i+1]
            HH[i+3, i+1] = 0
        end
        rmul!(view(HH, 1:min(i + 3, iend), :), G')
        # mul!(G, τ)
    end
    return HH
end

function doubleShiftQR!(
    HH::StridedMatrix,
    τ::Rotation,
    shiftTrace::Number,
    shiftDeterminant::Number,
    istart::Integer,
    iend::Integer,
)
    m = size(HH, 1)
    H11 = HH[istart, istart]
    H21 = HH[istart+1, istart]
    Htmp11 = HH[istart+2, istart]
    HH[istart+2, istart] = 0
    if istart + 3 <= m
        Htmp21 = HH[istart+3, istart]
        HH[istart+3, istart] = 0
        Htmp22 = HH[istart+3, istart+1]
        HH[istart+3, istart+1] = 0
    else
        # values don't matter in this case but variables should be initialized
        Htmp21 = Htmp22 = Htmp11
    end
    G1, r = givens(
        H11 * H11 + HH[istart, istart+1] * H21 - shiftTrace * H11 + shiftDeterminant,
        H21 * (H11 + HH[istart+1, istart+1] - shiftTrace),
        istart,
        istart + 1,
    )
    G2, _ = givens(r, H21 * HH[istart+2, istart+1], istart, istart + 2)
    vHH = view(HH, :, istart:m)
    lmul!(G1, vHH)
    lmul!(G2, vHH)
    vHH = view(HH, 1:min(istart + 3, m), :)
    rmul!(vHH, G1')
    rmul!(vHH, G2')
    lmul!(G1, τ)
    lmul!(G2, τ)
    for i = istart:iend-2
        for j = 1:2
            if i + j + 1 > iend
                break
            end
            # G, _ = givens(H.H,i+1,i+j+1,i)
            G, _ = givens(HH[i+1, i], HH[i+j+1, i], i + 1, i + j + 1)
            lmul!(G, view(HH, :, i:m))
            HH[i+j+1, i] = Htmp11
            Htmp11 = Htmp21
            # if i + j + 2 <= iend
            # Htmp21 = HH[i + j + 2, i + 1]
            # HH[i + j + 2, i + 1] = 0
            # end
            if i + 4 <= iend
                Htmp22 = HH[i+4, i+j]
                HH[i+4, i+j] = 0
            end
            rmul!(view(HH, 1:min(i + j + 2, iend), :), G')
            # mul!(G, τ)
        end
    end
    return HH
end

_eigvals!(A::StridedMatrix; kwargs...) = _eigvals!(_schur!(A; kwargs...))
_eigvals!(H::HessenbergMatrix; kwargs...) = _eigvals!(_schur!(H; kwargs...))
_eigvals!(H::HessenbergFactorization; kwargs...) = _eigvals!(_schur!(H; kwargs...))

function LinearAlgebra.eigvals!(
    A::StridedMatrix;
    sortby::Union{Function,Nothing} = LinearAlgebra.eigsortby,
    kwargs...,
)

    if ishermitian(A)
        return LinearAlgebra.sorteig!(eigvals!(Hermitian(A)), sortby)
    end
    LinearAlgebra.sorteig!(_eigvals!(A; kwargs...), sortby)
end

LinearAlgebra.eigvals!(
    H::HessenbergMatrix;
    sortby::Union{Function,Nothing} = LinearAlgebra.eigsortby,
    kwargs...,
) = LinearAlgebra.sorteig!(_eigvals!(H; kwargs...), sortby)

LinearAlgebra.eigvals!(
    H::HessenbergFactorization;
    sortby::Union{Function,Nothing} = LinearAlgebra.eigsortby,
    kwargs...,
) = LinearAlgebra.sorteig!(_eigvals!(H; kwargs...), sortby)

# To compute the eigenvalue of the pseudo triangular Schur matrix we just return
# the values of the 1x1 diagonal blocks and compute the eigenvalues of the 2x2
# blocks on the fly. We can check for exact zeros in the subdiagonal because
# the QR algorithm sets the elements to zero when a split happens
function _eigvals!(S::Schur{T}; tol = eps(real(T))) where {T}
    HH = S.data
    n = size(HH, 1)
    vals = Vector{complex(T)}(undef, n)
    i = 1
    while i < n
        Hii = HH[i, i]
        if iszero(HH[i+1, i])
            # 1x1 block
            vals[i] = Hii
            i += 1
        else
            # 2x2 block
            Hi1i1 = HH[i+1, i+1]
            d = Hii * Hi1i1 - HH[i, i+1] * HH[i+1, i]
            t = Hii + Hi1i1
            x = 0.5 * t
            y = sqrt(complex(x * x - d))
            vals[i] = x + y
            vals[i+1] = x - y
            i += 2
        end
    end
    if i == n
        vals[i] = HH[n, n]
    end
    return vals
end
