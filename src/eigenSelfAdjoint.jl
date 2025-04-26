using Printf
using LinearAlgebra
using LinearAlgebra: givensAlgorithm

## EigenQ
struct EigenQ{T}
    uplo::Char
    factors::Matrix{T}
    τ::Vector{T}
end

Base.size(Q::EigenQ) = (size(Q.factors, 1), size(Q.factors, 1))
function Base.size(Q::EigenQ, i::Integer)
    if i < 1
        throw(ArgumentError(""))
    elseif i < 3
        return size(Q.factors, 1)
    else
        return 1
    end
end

function LinearAlgebra.lmul!(Q::EigenQ, B::StridedVecOrMat)
    m, n = size(B)
    if size(Q, 1) != m
        throw(DimensionMismatch("first dimension of second argument matrix doesn't match the size of the first argument reflectors"))
    end
    if Q.uplo == 'L'
        for k = length(Q.τ):-1:1
            h = view(Q.factors, (k + 1):m, k)
            τ = Q.τ[k]
            tmp = h[1]
            h[1] = 1
            for j = 1:n
                tmp = τ * (h' * view(B, (k + 1):m, j))
                for i = 1:(m - k)
                    B[k + i, j] -= h[i] * tmp
                end
            end
            h[1] = tmp
        end
    elseif Q.uplo == 'U'
        for k = length(Q.τ):-1:1
            h = view(Q.factors, 1:(m - k), m - k + 1)
            τ = Q.τ[k]
            tmp = h[end]
            h[end] = 1
            for j = 1:n
                tmp = τ * (h' * view(B, 1:(n - k), j))
                for i = 1:(n - k)
                    B[i, j] -= h[i] * tmp
                end
            end
            h[end] = tmp
        end
    else
        throw(ArgumentError("Q.uplo is neither 'L' or 'U'. This should never happen."))
    end
    return B
end

function LinearAlgebra.rmul!(A::StridedMatrix, Q::EigenQ)
    m, n = size(A)
    if size(Q, 1) != n
        throw(DimensionMismatch("second dimension of first argument matrix doesn't match the size of the second argument reflectors"))
    end
    if Q.uplo == 'L'
        for k = 1:length(Q.τ)
            h = view(Q.factors, (k + 1):n, k)
            τ = Q.τ[k]
            tmp = h[1]
            h[1] = 1
            for i = 1:m
                tmp = transpose(view(A, i, (k + 1):n)) * h * τ
                for j = 1:(n - k)
                    A[i, k + j] -= tmp * h[j]'
                end
            end
            h[1] = tmp
        end
    elseif Q.uplo == 'U'
        for k = 1:length(Q.τ)
            h = view(Q.factors, 1:(n - k), n - k + 1)
            τ = Q.τ[k]
            tmp = h[end]
            h[end] = 1
            for i = 1:m
                tmp = transpose(view(A, i, 1:(n - k))) * h * τ
                for j = 1:(n - k)
                    A[i, j] -= tmp * h[j]'
                end
            end
            h[end] = tmp
        end
    else
        throw(ArgumentError("Q.uplo is neither 'L' or 'U'. This should never happen."))
    end
    return A
end

Base.Array(Q::EigenQ{T}) where T = lmul!(Q, Matrix{T}(I, size(Q, 1), size(Q, 1)))


## SymmetricTridiagonalFactorization
struct SymmetricTridiagonalFactorization{T,Treal,S} <: Factorization{T}
    reflectors::EigenQ{T}
    diagonals::SymTridiagonal{Treal,S}
end

Base.size(S::SymmetricTridiagonalFactorization, i::Integer) = size(S.reflectors.factors, i)

function Base.getproperty(S::SymmetricTridiagonalFactorization, s::Symbol)
    if s == :Q
        return S.reflectors
    else
        return getfield(S, s)
    end
end

## Eigen solvers
function _updateVectors!(c, s, j, vectors)
    @inbounds for i = 1:size(vectors, 1)
        v1 = vectors[i, j+1]
        v2 = vectors[i, j]
        vectors[i, j+1] = c * v1 + s * v2
        vectors[i, j] = c * v2 - s * v1
    end
end


function eigvals2x2(d1::Number, d2::Number, e::Number)
    r1 = (d1 + d2) / 2
    r2 = hypot(d1 - d2, 2 * e) / 2
    return r1 + r2, r1 - r2
end
function eig2x2!(d::StridedVector, e::StridedVector, j::Integer, vectors::Matrix)
    dj = d[j]
    dj1 = d[j + 1]
    ej = e[j]
    r1 = (dj + dj1) / 2
    r2 = hypot(dj - dj1, 2 * ej) / 2
    λ⁺ = r1 + r2
    λ⁻ = r1 - r2
    d[j] = λ⁺
    d[j + 1] = λ⁻
    e[j] = 0
    if iszero(ej)
        c = one(λ⁺)
        s = zero(λ⁺)
    elseif abs(λ⁺ - dj) > abs(λ⁻ - dj)
        c = -ej / hypot(ej, λ⁺ - dj)
        s = sqrt(1 - c*c)
    else
        s = abs(ej) / hypot(ej, λ⁻ - dj)
        c = copysign(sqrt(1 - s*s), -ej)
    end

    _updateVectors!(c, s, j, vectors)

    return c, s
end

function eigvalsPWK!(S::SymTridiagonal{T}; tol = eps(T)) where {T<:Real}
    d = S.dv
    e = S.ev
    n = length(d)
    blockstart = 1
    blockend = n
    iter = 0
    @inbounds begin
        for i = 1:n-1
            e[i] = abs2(e[i])
        end
        while true
            # Check for zero off diagonal elements
            for be = (blockstart + 1):n
                if abs(e[be - 1]) <= tol * sqrt(abs(d[be - 1])) * sqrt(abs(d[be]))
                    blockend = be - 1
                    break
                end
                blockend = n
            end

            # @debug "submatrix" blockstart blockend

            # Deflate?
            if blockstart == blockend
                # @debug "Deflate? Yes!"
                blockstart += 1
            elseif blockstart + 1 == blockend
                # @debug "Defalte? Yes, but after solving 2x2 block!"
                d[blockstart], d[blockend] =
                    eigvals2x2(d[blockstart], d[blockend], sqrt(e[blockstart]))
                e[blockstart] = zero(T)
                blockstart += 1
            else
                # @debug "Deflate? No!"
                # Calculate shift
                sqrte = sqrt(e[blockstart])
                μ = (d[blockstart+1] - d[blockstart]) / (2 * sqrte)
                r = hypot(μ, one(T))
                μ = d[blockstart] - sqrte / (μ + copysign(r, μ))

                # QL bulk chase
                # @debug "Values before PWK QL bulge chase" e[blockstart] e[blockend-1] μ
                singleShiftQLPWK!(S, μ, blockstart, blockend)

                rotations = blockend - blockstart
                iter += rotations
                # @debug "Values after PWK QL bulge chase" e[blockstart] e[blockend-1] rotations
            end
            if blockstart >= n
                break
            end
        end
    end

    return d
end

function eigQL!(
    S::SymTridiagonal{T};
    vectors::Matrix = zeros(T, 0, size(S, 1)),
    tol = eps(T),
) where {T<:Real}
    d = S.dv
    e = S.ev
    n = length(d)

    if size(vectors, 2) != n
        throw(
            DimensionMismatch(
                "eigenvector matrix must have $(n) columns but had $(size(vectors, 2))",
            ),
        )
    end
    if size(vectors, 1) > n
        throw(
            DimensionMismatch(
                "eigenvector matrix must have at most $(n) rows but had $(size(vectors, 1))",
            ),
        )
    end

    blockstart = 1
    blockend = n
    @inbounds begin
        while true
            # Check for zero off diagonal elements
            for be = (blockstart + 1):n
                if abs(e[be - 1]) <= tol * sqrt(abs(d[be - 1])) * sqrt(abs(d[be]))
                    blockend = be - 1
                    break
                end
                blockend = n
            end

            # @debug "submatrix" blockstart blockend

            # Deflate?
            if blockstart == blockend
                # @debug "Deflate? Yes!"
                blockstart += 1
            elseif blockstart + 1 == blockend
                # @debug "Deflate? Yes, but after solving 2x2 block"
                eig2x2!(d, e, blockstart, vectors)
                blockstart += 2
            else
                # @debug "Deflate? No!"
                # Calculate shift
                μ = (d[blockstart + 1] - d[blockstart]) / (2 * e[blockstart])
                r = hypot(μ, one(T))
                μ = d[blockstart] - (e[blockstart] / (μ + copysign(r, μ)))

                # QL bulk chase
                # @debug "Values before bulge chase" e[blockstart] e[blockend - 1] d[blockstart] μ
                singleShiftQL!(S, μ, blockstart, blockend, vectors)
                # @debug "Values after bulge chase" e[blockstart] e[blockend - 1] d[blockstart]
            end

            if blockstart >= n
                break
            end
        end
    end

    return d, vectors
end

function eigQR!(
    S::SymTridiagonal{T};
    vectors::Matrix = zeros(T, 0, size(S, 1)),
    tol = eps(T),
) where {T<:Real}
    d = S.dv
    e = S.ev
    n = length(d)
    blockstart = 1
    blockend = n
    @inbounds begin
        while true
            # Check for zero off diagonal elements
            for be = (blockstart + 1):n
                if abs(e[be - 1]) <= tol * sqrt(abs(d[be - 1])) * sqrt(abs(d[be]))
                    blockend = be - 1
                    break
                end
                blockend = n
            end

            # @debug "submatrix" blockstart blockend

            # Deflate?
            if blockstart == blockend
                # @debug "Deflate? Yes!"
                blockstart += 1
            elseif blockstart + 1 == blockend
                # @debug "Deflate? Yes, but after solving 2x2 block!"
                eig2x2!(d, e, blockstart, vectors)
                blockstart += 2
            else
                # @debug "Deflate? No!"
                # Calculate shift
                μ = (d[blockend - 1] - d[blockend]) / (2 * e[blockend - 1])
                r = hypot(μ, one(T))
                μ = d[blockend] - (e[blockend - 1] / (μ + copysign(r, μ)))

                # QR bulk chase
                # @debug "Values before QR bulge chase" e[blockstart] e[blockend - 1] d[blockend] μ
                singleShiftQR!(S, μ, blockstart, blockend, vectors)
                # @debug "Values after QR bulge chase" e[blockstart] e[blockend - 1] d[blockend]
            end
            if blockstart >= n
                break
            end
        end
    end

    return d, vectors
end

# Own implementation based on Parlett's book
function singleShiftQLPWK!(
    S::SymTridiagonal,
    shift::Number,
    istart::Integer = 1,
    iend::Integer = length(S.dv),
)
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
        if i < iend - 1
            e[i+1] = s * ζ
        end
        ci1 = ci
        ci = π / ζ
        s = ei / ζ
        di = d[i]
        γi1 = γi
        γi = ci * (di - shift) - s * γi1
        d[i+1] = γi1 + di - γi
        π = ci == 0 ? ei * ci1 : γi * γi / ci
    end
    e[istart] = s * π
    d[istart] = shift + γi
    S
end

# Own implementation based on Parlett's book
function singleShiftQL!(
    S::SymTridiagonal,
    shift::Number,
    istart::Integer = 1,
    iend::Integer = length(S.dv),
    vectors = zeros(eltype(S), 0, size(S, 1)),
)
    d, e = S.dv, S.ev
    n = length(d)
    γi = d[iend] - shift
    π = γi
    ci, si = one(eltype(S)), zero(eltype(S))
    @inbounds for i = (iend-1):-1:istart
        ei = e[i]
        ci1 = ci
        si1 = si
        ci, si, ζ = givensAlgorithm(π, ei)
        if i < iend - 1
            e[i+1] = si1 * ζ
        end
        di = d[i]
        γi1 = γi
        γi = ci * ci * (di - shift) - si * si * γi1
        d[i+1] = γi1 + di - γi
        π = ci == 0 ? -ei * ci1 : γi / ci

        # update eigen vectors
        _updateVectors!(ci, si, i, vectors)
    end
    e[istart] = si * π
    d[istart] = shift + γi
    S
end

# Own implementation based on Parlett's book
function singleShiftQR!(
    S::SymTridiagonal,
    shift::Number,
    istart::Integer = 1,
    iend::Integer = length(S.dv),
    vectors = zeros(eltype(S), 0, size(S, 1)),
)
    d, e = S.dv, S.ev
    n = length(d)
    γi = d[istart] - shift
    π = γi
    ci, si = one(eltype(S)), zero(eltype(S))
    for i = (istart+1):iend
        ei = e[i-1]
        ci1 = ci
        si1 = si
        ci, si, ζ = givensAlgorithm(π, ei)
        if i > istart + 1
            e[i-2] = si1 * ζ
        end
        di = d[i]
        γi1 = γi
        γi = ci * ci * (di - shift) - si * si * γi1
        d[i-1] = γi1 + di - γi
        π = ci == 0 ? -ei * ci1 : γi / ci

        # update eigen vectors
        _updateVectors!(ci, -si, i - 1, vectors)
    end
    e[iend-1] = si * π
    d[iend] = shift + γi
    S
end

symtri!(A::Hermitian) = A.uplo == 'L' ? symtriLower!(A.data) : symtriUpper!(A.data)
symtri!(A::Symmetric{<:Real}) = A.uplo == 'L' ? symtriLower!(A.data) : symtriUpper!(A.data)

# Assume that lower triangle stores the relevant part
function symtriLower!(
    AS::StridedMatrix{T},
    τ = zeros(T, size(AS, 1) - 1),
    u = Vector{T}(undef, size(AS, 1)),
) where {T}
    n = size(AS, 1)

    # We ignore any non-real components of the diagonal
    @inbounds for i = 1:n
        AS[i, i] = real(AS[i, i])
    end

    @inbounds for k = 1:(n-2+!(T <: Real))

        x = view(AS, (k + 1):n, k)

        τk = LinearAlgebra.reflector!(x)
        τ[k] = τk

        # Temporarily store the implicit 1
        tmp = x[1]
        x[1] = 1

        Ã = view(AS, (k + 1):n, (k + 1):n)
        ũ = view(u, (k + 1):n)

        # Form Ãvτ
        mul!(ũ, Hermitian(Ã, :L), x, τk, zero(τk))

        # Form τx'Ãvτ (which is real except for rounding)
        ξ = real(τk'*(x'*ũ))

        # Compute the rank up and down grade
        # Ã = Ã + τx'Ãvτ - τx'Ã - Ãvτ
        for j in 1:(n - k)
            xⱼ = x[j]
            ũⱼ = ũ[j]
            ξxⱼ = ξ * xⱼ
            for i in j:(n - k)
                AS[k + i, k + j] += x[i] * ξxⱼ' - x[i] * ũⱼ' - ũ[i] * xⱼ'
            end
        end

        # Restore element
        x[1] = tmp
    end
    SymmetricTridiagonalFactorization(
        EigenQ(
            'L',
            AS,
            τ,
        ),
        SymTridiagonal(real(diag(AS)), real(diag(AS, -1))),
    )
end

# Assume that upper triangle stores the relevant part
function symtriUpper!(
    AS::StridedMatrix{T},
    τ = zeros(T, size(AS, 1) - 1),
    u = Vector{T}(undef, size(AS, 1)),
) where {T}
    n = LinearAlgebra.checksquare(AS)

    # We ignore any non-real components of the diagonal
    @inbounds for i = 1:n
        AS[i, i] = real(AS[i, i])
    end

    @inbounds for k = 1:(n-2+!(T <: Real))
        # LAPACK allows the reflection element to be chosen freely whereas
        # LinearAlgebra's reflector! assumes it is the first element in the vector
        # Maybe we should implement a method similar to the LAPACK version
        x = view(AS, 1:(n - k), n - k + 1)
        xᵣ = view(AS, (n - k):-1:1, n - k + 1)

        τk = LinearAlgebra.reflector!(xᵣ)
        τ[k] = τk

        # Temporarily store the implicit 1
        tmp = x[end]
        x[end] = 1

        Ã = view(AS, 1:(n - k), 1:(n - k))
        ũ = view(u, 1:(n - k))

        # Form Ãvτ
        mul!(ũ, Hermitian(Ã, :U), x, τk, zero(τk))

        # Form τx'Ãvτ (which is real except for rounding)
        ξ = real(τk'*(x'*ũ))

        # Compute the rank up and down grade
        # Ã = Ã + τx'Ãvτ - τx'Ã - Ãvτ
        for j in 1:(n - k)
            xⱼ = x[j]
            ũⱼ = ũ[j]
            ξxⱼ = ξ * xⱼ
            for i in 1:j
                AS[i, j] += x[i] * ξxⱼ' - x[i] * ũⱼ' - ũ[i] * xⱼ'
            end
        end

        # Restore element
        x[end] = tmp
    end
    SymmetricTridiagonalFactorization(
        EigenQ(
            'U',
            AS,
            τ,
        ),
        SymTridiagonal(real(diag(AS)), real(diag(AS, 1))),
    )
end

_eigvals!(A::SymTridiagonal; tol = eps(real(eltype(A))), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) =
    LinearAlgebra.sorteig!(eigvalsPWK!(A; tol), sortby == nothing ? LinearAlgebra.eigsortby : sortby)

_eigvals!(A::SymmetricTridiagonalFactorization; tol = eps(real(eltype(A))), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) =
    _eigvals!(A.diagonals; tol, sortby)

_eigvals!(A::Hermitian; tol = eps(real(eltype(A))), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) =
    _eigvals!(symtri!(A); tol, sortby)

_eigvals!(A::Symmetric{<:Real}; tol = eps(eltype(A)), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) =
    _eigvals!(symtri!(A); tol, sortby)

LinearAlgebra.eigvals!(A::SymmetricTridiagonalFactorization; tol = eps(real(eltype(A))), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) =
    _eigvals!(A; tol, sortby)

LinearAlgebra.eigvals!(A::SymTridiagonal; tol = eps(real(eltype(A))), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) =
    _eigvals!(A; tol, sortby)

LinearAlgebra.eigvals!(A::Hermitian; tol = eps(real(eltype(A))), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby, alg = nothing) =
    _eigvals!(A; tol, sortby)

LinearAlgebra.eigvals!(A::Symmetric{<:Real}; tol = eps(eltype(A)), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby, alg = nothing) =
    _eigvals!(A; tol, sortby)

_eigen!(A::SymmetricTridiagonalFactorization; tol = eps(real(eltype(A))), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) =
    LinearAlgebra.Eigen(
        LinearAlgebra.sorteig!(
            eigQL!(
                A.diagonals;
                vectors = Array(A.Q),
                tol
            )...,
            sortby == nothing ? LinearAlgebra.eigsortby : sortby
        )...
    )

_eigen!(A::SymTridiagonal; tol = eps(real(eltype(A))), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) =
    LinearAlgebra.Eigen(
        LinearAlgebra.sorteig!(
            eigQL!(
                A;
                vectors = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)),
                tol
            )...,
            sortby == nothing ? LinearAlgebra.eigsortby : sortby
        )...
    )

_eigen!(A::Hermitian; tol = eps(real(eltype(A))), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) =
    _eigen!(symtri!(A); tol, sortby)

_eigen!(A::Symmetric{<:Real}; tol = eps(eltype(A)), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) =
    _eigen!(symtri!(A); tol, sortby)

LinearAlgebra.eigen!(A::SymmetricTridiagonalFactorization; tol = eps(real(eltype(A))), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) =
    _eigen!(A; tol, sortby)

LinearAlgebra.eigen!(A::SymTridiagonal; tol = eps(real(eltype(A))), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby) =
    _eigen!(A; tol, sortby)

LinearAlgebra.eigen!(A::Hermitian; tol = eps(real(eltype(A))), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby, alg = nothing) =
    _eigen!(A; tol, sortby)

LinearAlgebra.eigen!(A::Symmetric{<:Real}; tol = eps(eltype(A)), sortby::Union{Function,Nothing}=LinearAlgebra.eigsortby, alg = nothing) =
    _eigen!(A; tol, sortby)


function eigen2!(
    A::SymmetricTridiagonalFactorization;
    tol = eps(real(float(one(eltype(A))))),
)
    V = zeros(eltype(A), 2, size(A, 1))
    V[1] = 1
    V[end] = 1
    LinearAlgebra.sorteig!(
        eigQL!(A.diagonals; vectors = rmul!(V, A.Q), tol)...,
        LinearAlgebra.eigsortby
    )
end

function eigen2!(A::SymTridiagonal; tol = eps(real(float(one(eltype(A))))))
    V = zeros(eltype(A), 2, size(A, 1))
    V[1] = 1
    V[end] = 1
    LinearAlgebra.sorteig!(
        eigQL!(A; vectors = V, tol)...,
        LinearAlgebra.eigsortby
    )
end

eigen2!(A::Hermitian; tol = eps(float(real(one(eltype(A)))))) =
    eigen2!(symtri!(A); tol)

eigen2!(A::Symmetric{<:Real}; tol = eps(float(one(eltype(A))))) =
    eigen2!(symtri!(A); tol)


eigen2(A::SymTridiagonal; tol = eps(float(real(one(eltype(A)))))) =
    eigen2!(copy(A); tol)

eigen2(A::Hermitian; tol = eps(float(real(one(eltype(A)))))) = eigen2!(copy(A); tol)

eigen2(A::Symmetric{<:Real}; tol = eps(float(one(eltype(A))))) = eigen2!(copy(A); tol)

# First method of each type here is identical to the method defined in
# LinearAlgebra but is needed for disambiguation
const _eigencopy_oftype = if VERSION >= v"1.9"
    LinearAlgebra.eigencopy_oftype
else
    LinearAlgebra.copy_oftype
end

if VERSION < v"1.7"
    function LinearAlgebra.eigvals(A::Hermitian{<:Real})
        T = typeof(sqrt(zero(eltype(A))))
        return eigvals!(_eigencopy_oftype(A, T))
    end
    function LinearAlgebra.eigvals(A::Hermitian{<:Complex})
        T = typeof(sqrt(zero(eltype(A))))
        return eigvals!(_eigencopy_oftype(A, T))
    end
    function LinearAlgebra.eigvals(A::Hermitian)
        T = typeof(sqrt(zero(eltype(A))))
        return eigvals!(_eigencopy_oftype(A, T))
    end
    function LinearAlgebra.eigen(A::Hermitian{<:Real})
        T = typeof(sqrt(zero(eltype(A))))
        return eigen!(_eigencopy_oftype(A, T))
    end
    function LinearAlgebra.eigen(A::Hermitian{<:Complex})
        T = typeof(sqrt(zero(eltype(A))))
        return eigen!(_eigencopy_oftype(A, T))
    end
    function LinearAlgebra.eigen(A::Hermitian)
        T = typeof(sqrt(zero(eltype(A))))
        return eigen!(_eigencopy_oftype(A, T))
    end
end

# Aux (should go somewhere else at some point)
function LinearAlgebra.givensAlgorithm(f::Real, g::Real)
    h = hypot(f, g)
    return f / h, g / h, h
end
