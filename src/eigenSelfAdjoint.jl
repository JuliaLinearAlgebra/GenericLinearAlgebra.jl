using Printf
using LinearAlgebra
using LinearAlgebra: givensAlgorithm

struct SymmetricTridiagonalFactorization{T} <: Factorization{T}
    uplo::Char
    factors::Matrix{T}
    τ::Vector{T}
    diagonals::SymTridiagonal
end

Base.size(S::SymmetricTridiagonalFactorization, i::Integer) = size(S.factors, i)

struct EigenQ{T} <: AbstractMatrix{T}
    uplo::Char
    factors::Matrix{T}
    τ::Vector{T}
end

function Base.getproperty(S::SymmetricTridiagonalFactorization, s::Symbol)
    if s == :Q
        return EigenQ(S.uplo, S.factors, S.τ)
    else
        return getfield(S, s)
    end
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
        throw(DimensionMismatch(""))
    end
    if Q.uplo == 'L'
        for k in length(Q.τ):-1:1
            for j in 1:size(B, 2)
                b = view(B, :, j)
                vb = B[k + 1, j]
                for i in (k + 2):m
                    vb += Q.factors[i, k]'B[i, j]
                end
                τkvb = Q.τ[k]'vb
                B[k + 1, j] -= τkvb
                for i in (k + 2):m
                    B[i, j] -= Q.factors[i, k]*τkvb
                end
            end
        end
    elseif Q.uplo == 'U'
        for k in length(Q.τ):-1:1
            for j in 1:size(B, 2)
                b = view(B, :, j)
                vb = B[k + 1, j]
                for i in (k + 2):m
                    vb += Q.factors[k, i]*B[i, j]
                end
                τkvb = Q.τ[k]'vb
                B[k + 1, j] -= τkvb
                for i in (k + 2):m
                    B[i, j] -= Q.factors[k, i]'*τkvb
                end
            end
        end
    else
        throw(ArgumentError("Q.uplo is neither 'L' or 'U'. This should never happen."))
    end
    return B
end

function LinearAlgebra.rmul!(A::StridedMatrix, Q::EigenQ)
    m, n = size(A)
    if size(Q, 1) != n
        throw(DimensionMismatch(""))
    end
    if Q.uplo == 'L'
        for k in 1:length(Q.τ)
            for i in 1:size(A, 1)
                a = view(A, i, :)
                va = A[i, k + 1]
                for j in (k + 2):n
                    va += A[i, j]*Q.factors[j, k]
                end
                τkva = va*Q.τ[k]'
                A[i, k + 1] -= τkva
                for j in (k + 2):n
                    A[i, j] -= τkva*Q.factors[j, k]'
                end
            end
        end
    elseif Q.uplo == 'U' # FixMe! Should consider reordering loops
        for k in 1:length(Q.τ)
            for i in 1:size(A, 1)
                a = view(A, i, :)
                va = A[i, k + 1]
                for j in (k + 2):n
                    va += A[i, j]*Q.factors[k, j]'
                end
                τkva = va*Q.τ[k]'
                A[i, k + 1] -= τkva
                for j in (k + 2):n
                    A[i, j] -= τkva*Q.factors[k, j]
                end
            end
        end
    else
        throw(ArgumentError("Q.uplo is neither 'L' or 'U'. This should never happen."))
    end
    return A
end

Base.Array(Q::EigenQ) = lmul!(Q, Matrix{eltype(Q)}(I, size(Q, 1), size(Q, 1)))

function _updateVectors!(c, s, j, vectors)
    @inbounds for i = 1:size(vectors, 1)
        v1 = vectors[i, j + 1]
        v2 = vectors[i, j]
        vectors[i, j + 1] = c*v1 + s*v2
        vectors[i, j]     = c*v2 - s*v1
    end
end


function eigvals2x2(d1::Number, d2::Number, e::Number)
    r1 = (d1 + d2)/2
    r2 = hypot(d1 - d2, 2*e)/2
    return r1 + r2, r1 - r2
end
function eigQL2x2!(d::StridedVector, e::StridedVector, j::Integer, vectors::Matrix)
    dj = d[j]
    dj1 = d[j + 1]
    ej = e[j]
    r1 = (dj + dj1)/2
    r2 = hypot(dj - dj1, 2*ej)/2
    λ = r1 + r2
    d[j] = λ
    d[j + 1] = r1 - r2
    e[j] = 0
    c = ej/(dj - λ)
    if isfinite(c) # FixMe! is this the right fix for overflow?
        h = hypot(one(c), c)
        c /= h
        s = inv(h)
    else
        c = one(c)
        s = zero(c)
    end

    _updateVectors!(c, s, j, vectors)

    return c, s
end

function eigvalsPWK!(S::SymTridiagonal{T}; tol = eps(T), debug::Bool=false) where T<:Real
    d = S.dv
    e = S.ev
    n = length(d)
    blockstart = 1
    blockend = n
    iter = 0
    @inbounds begin
        for i = 1:n - 1
            e[i] = abs2(e[i])
        end
        while true
            # Check for zero off diagonal elements
            for be in blockstart + 1:n
                if abs(e[be - 1]) < tol*sqrt(abs(d[be - 1]))*sqrt(abs(d[be]))
                    blockend = be - 1
                    break
                end
                blockend = n
            end

            # Deflate?
            if blockstart == blockend
                # Yes
                blockstart += 1
            elseif blockstart + 1 == blockend
                debug && println("Yes, but after sreolving 2x2 block")
                d[blockstart], d[blockend] = eigvals2x2(d[blockstart], d[blockend], sqrt(e[blockstart]))
                e[blockstart] = zero(T)
                blockstart += 1
            else
                # Calculate shift
                sqrte = sqrt(e[blockstart])
                μ = (d[blockstart + 1] - d[blockstart])/(2*sqrte)
                r = hypot(μ, one(T))
                μ = d[blockstart] - sqrte/(μ + copysign(r, μ))

                # QL bulk chase
                singleShiftQLPWK!(S, μ, blockstart, blockend)

                debug && @printf("QL, blockstart: %d, blockend: %d, e[blockstart]: %e, e[blockend-1]:%e, μ: %e, rotations: %d\n", blockstart, blockend, e[blockstart], e[blockend-1], μ, iter += blockend - blockstart)
            end
            if blockstart >= n
                break
            end
        end
    end
    sort!(d)
end

function eigQL!(S::SymTridiagonal{T};
                vectors::Matrix = zeros(T, 0, size(S, 1)),
                tol = eps(T),
                debug::Bool=false) where T<:Real
    d = S.dv
    e = S.ev
    n = length(d)

    if size(vectors, 2) != n
        throw(DimensionMismatch("eigenvector matrix must have $(n) columns but had $(size(vectors, 2))"))
    end
    if size(vectors, 1) > n
        throw(DimensionMismatch("eigenvector matrix must have at most $(n) rows but had $(size(vectors, 1))"))
    end

    blockstart = 1
    blockend = n
    @inbounds begin
        while true
            # Check for zero off diagonal elements
            for be = blockstart+1:n
                if abs(e[be-1]) < tol*sqrt(abs(d[be-1]))*sqrt(abs(d[be]))
                    blockend = be - 1
                    break
                end
                blockend = n
            end
            # Deflate?
            if blockstart == blockend
                # Yes!
                blockstart += 1
            elseif blockstart + 1 == blockend
                debug && println("Yes, but after solving 2x2 block")
                eigQL2x2!(d, e, blockstart, vectors)
                blockstart += 1
            else
                # Calculate shift
                μ = (d[blockstart + 1] - d[blockstart])/(2*e[blockstart])
                r = hypot(μ, one(T))
                μ = d[blockstart] - (e[blockstart]/(μ + copysign(r, μ)))

                # QL bulk chase
                singleShiftQL!(S, μ, blockstart, blockend, vectors)
                debug && @printf("QL, blockstart: %d, blockend: %d, e[blockstart]: %e, e[blockend-1]:%e, μ: %f\n", blockstart, blockend, e[blockstart], e[blockend-1], μ)
            end
            if blockstart >= n
                break
            end
        end
    end
    p = sortperm(d)
    return d[p], vectors[:,p]
end

function eigQR!(S::SymTridiagonal{T};
                vectors::Matrix = zeros(T, 0, size(S, 1)),
                tol = eps(T),
                debug::Bool=false) where T<:Real
    d = S.dv
    e = S.ev
    n = length(d)
    blockstart = 1
    blockend = n
    @inbounds begin
        while true
            # Check for zero off diagonal elements
            for be in (blockstart + 1):n
                if abs(e[be - 1]) <= tol*sqrt(abs(d[be - 1]))*sqrt(abs(d[be]))
                    blockend = be - 1
                    break
                end
                blockend = n
            end
            # Deflate?
            if blockstart == blockend
                # Yes!
                blockstart += 1
            elseif blockstart + 1 == blockend
                debug && println("Yes, but after solving 2x2 block")
                eigQL2x2!(d, e, blockstart, vectors)
                blockstart += 1
            else
                # Calculate shift
                μ = (d[blockend - 1] - d[blockend])/(2*e[blockend - 1])
                r = hypot(μ, one(T))
                μ = d[blockend] - (e[blockend - 1]/(μ + copysign(r, μ)))

                # QR bulk chase
                singleShiftQR!(S, μ, blockstart, blockend, vectors)

                debug && @printf("QR, blockstart: %d, blockend: %d, e[blockstart]: %e, e[blockend-1]:%e, d[blockend]: %f, μ: %f\n", blockstart, blockend, e[blockstart], e[blockend-1], d[blockend], μ)
            end
            if blockstart >= n
                break
            end
        end
    end
    p = sortperm(d)
    return d[p], vectors[:,p]
end

# Own implementation based on Parlett's book
function singleShiftQLPWK!(S::SymTridiagonal,
                           shift::Number,
                           istart::Integer = 1,
                           iend::Integer = length(S.dv))
    d = S.dv
    e = S.ev
    n = length(d)
    γi = d[iend] - shift
    π = abs2(γi)
    ci = one(eltype(S))
    s = zero(eltype(S))
    @inbounds for i = iend-1:-1:istart
        ei = e[i]
        ζ  = π + ei
        if i < iend-1
            e[i+1] = s*ζ
        end
        ci1    = ci
        ci     = π/ζ
        s      = ei/ζ
        di     = d[i]
        γi1    = γi
        γi     = ci*(di - shift) - s*γi1
        d[i+1] = γi1 + di - γi
        π      = ci == 0 ? ei*ci1 : γi*γi/ci
    end
    e[istart] = s*π
    d[istart] = shift + γi
    S
end

# Own implementation based on Parlett's book
function singleShiftQL!(S::SymTridiagonal,
                        shift::Number,
                        istart::Integer = 1,
                        iend::Integer = length(S.dv),
                        vectors = zeros(eltype(S), 0, size(S, 1)))
    d, e   = S.dv, S.ev
    n      = length(d)
    γi     = d[iend] - shift
    π      = γi
    ci, si = one(eltype(S)), zero(eltype(S))
    @inbounds for i = (iend - 1):-1:istart
        ei        = e[i]
        ci1       = ci
        si1       = si
        ci, si, ζ = givensAlgorithm(π, ei)
        if i < iend - 1
            e[i + 1] = si1*ζ
        end
        di       = d[i]
        γi1      = γi
        γi       = ci*ci*(di - shift) - si*si*γi1
        d[i + 1] = γi1 + di - γi
        π        = ci == 0 ? -ei*ci1 : γi/ci

        # update eigen vectors
        _updateVectors!(ci, si, i, vectors)
    end
    e[istart] = si*π
    d[istart] = shift + γi
    S
end

# Own implementation based on Parlett's book
function singleShiftQR!(S::SymTridiagonal,
                        shift::Number,
                        istart::Integer = 1,
                        iend::Integer = length(S.dv),
                        vectors = zeros(eltype(S), 0, size(S, 1)))
    d, e   = S.dv, S.ev
    n      = length(d)
    γi     = d[istart] - shift
    π      = γi
    ci, si = one(eltype(S)), zero(eltype(S))
    for i = (istart + 1):iend
        ei        = e[i - 1]
        ci1       = ci
        si1       = si
        ci, si, ζ = givensAlgorithm(π, ei)
        if i > istart + 1
            e[i - 2] = si1*ζ
        end
        di       = d[i]
        γi1      = γi
        γi       = ci*ci*(di - shift) - si*si*γi1
        d[i - 1] = γi1 + di - γi
        π        = ci == 0 ? -ei*ci1 : γi/ci

        # update eigen vectors
        _updateVectors!(ci, -si, i - 1, vectors)
    end
    e[iend - 1] = si*π
    d[iend]     = shift + γi
    S
end

function zeroshiftQR!(A::Bidiagonal{T}) where T
    d = A.dv
    e = A.ev
    n = length(d)
    oldc = one(T)
    olds = oldc
    c = oldc
    for i = 1:n-1
        c, s, r = givensAlgorithm(d[i]*c,e[i])
        if i > 1
            e[i-1] = olds*r
        end
        oldc, olds, d[i] = givensAlgorithm(oldc*r,d[i+1]*s)
    end
    h = d[n]*c
    e[n-1] = h*olds
    d[n] = h*oldc
    return A
end

symtri!(A::Hermitian) = A.uplo == 'L' ? symtriLower!(A.data) : symtriUpper!(A.data)
symtri!(A::Symmetric{T}) where {T<:Real} = A.uplo == 'L' ? symtriLower!(A.data) : symtriUpper!(A.data)

# Assume that lower triangle stores the relevant part
function symtriLower!(AS::StridedMatrix{T},
                      τ = zeros(T, size(AS, 1) - 1),
                      u = Vector{T}(undef, size(AS, 1))) where T
    n = size(AS, 1)

    # We ignore any non-real components of the diagonal
    @inbounds for i in 1:n
        AS[i,i] = real(AS[i,i])
    end

    @inbounds for k = 1:(n - 2 + !(T<:Real))

        τk = LinearAlgebra.reflector!(view(AS, (k + 1):n, k))
        τ[k] = τk

        for i in (k + 1):n
            u[i] = AS[i, k + 1]
        end
        for j in (k + 2):n
            ASjk = AS[j, k]
            for i in j:n
                u[i] += AS[i, j]*ASjk
            end
        end
        for j in (k + 1):(n - 1)
            tmp = zero(T)
            for i in j+1:n
                tmp += AS[i,j]'AS[i,k]
            end
            u[j] += tmp
        end

        vcAv = u[k + 1]
        for i in (k + 2):n
            vcAv += AS[i, k]'u[i]
        end
        ξτ2 = real(vcAv)*abs2(τk)/2

        u[k + 1] = u[k + 1]*τk - ξτ2
        for i in (k + 2):n
            u[i] = u[i]*τk - ξτ2*AS[i, k]
        end

        AS[k + 1, k + 1] -= 2real(u[k + 1])
        for i in (k + 2):n
            AS[i, k + 1] -= u[i] + AS[i, k]*u[k + 1]'
        end
        for j in (k + 2):n
            ASjk = AS[j, k]
            uj = u[j]
            AS[j,j] -= 2real(uj*ASjk')
            for i = (j + 1):n
                AS[i, j] -= u[i]*ASjk' + AS[i, k]*uj'
            end
        end
    end
    SymmetricTridiagonalFactorization('L', AS, τ, SymTridiagonal(real(diag(AS)), real(diag(AS, -1))))
end

# Assume that upper triangle stores the relevant part
function symtriUpper!(AS::StridedMatrix{T},
                      τ = zeros(T, size(AS, 1) - 1),
                      u = Vector{T}(undef, size(AS, 1))) where T
    n = LinearAlgebra.checksquare(AS)

    # We ignore any non-real components of the diagonal
    @inbounds for i in 1:n
        AS[i,i] = real(AS[i,i])
    end

    @inbounds for k = 1:(n - 2 + !(T<:Real))
        # This is a bit convoluted method to get the conjugated vector but conjugation is required for correctness of arrays of quaternions. Eventually, it should be sufficient to write vec(x') but it currently (July 10, 2018) hits a bug in LinearAlgebra
        τk = LinearAlgebra.reflector!(vec(transpose(view(AS, k, (k + 1):n)')))
        τ[k] = τk'

        for j in (k + 1):n
            tmp = AS[k + 1, j]
            for i in (k + 2):j
                tmp += AS[k, i]*AS[i, j]
            end
            for i in (j + 1):n
                tmp += AS[k, i]*AS[j, i]'
            end
            u[j] = τk'*tmp
        end

        vcAv = u[k + 1]
        for i in (k + 2):n
            vcAv += u[i]*AS[k, i]'
        end
        ξ = real(vcAv*τk)

        for j in (k + 1):n
            ujt       = u[j]
            hjt       = j > (k + 1) ? AS[k, j] : one(ujt)
            ξhjt      = ξ*hjt
            for i in (k + 1):(j - 1)
                hit = i > (k + 1) ? AS[k, i] : one(ujt)
                AS[i, j] -= hit'*ujt + u[i]'*hjt - hit'*ξhjt
            end
            AS[j, j] -= 2*real(hjt'*ujt) - abs2(hjt)*ξ
        end
    end
    SymmetricTridiagonalFactorization('U', AS, τ, SymTridiagonal(real(diag(AS)), real(diag(AS, 1))))
end


_eigvals!(A::SymmetricTridiagonalFactorization;
          tol = eps(real(eltype(A))),
          debug = false) = eigvalsPWK!(A.diagonals, tol = tol, debug = debug)

_eigvals!(A::SymTridiagonal;
          tol = eps(real(eltype(A))),
          debug = false) = eigvalsPWK!(A, tol = tol, debug = debug)

_eigvals!(A::Hermitian;
          tol = eps(real(eltype(A))),
          debug = false) = eigvals!(symtri!(A), tol = tol, debug = debug)


LinearAlgebra.eigvals!(A::SymmetricTridiagonalFactorization;
                       tol = eps(real(eltype(A))),
                       debug = false) = _eigvals!(A, tol = tol, debug = debug)

LinearAlgebra.eigvals!(A::SymTridiagonal;
                       tol = eps(real(eltype(A))),
                       debug = false) = _eigvals!(A, tol = tol, debug = debug)

LinearAlgebra.eigvals!(A::Hermitian;
                       tol = eps(real(eltype(A))),
                       debug = false) = _eigvals!(A, tol = tol, debug = debug)


_eigen!(A::SymmetricTridiagonalFactorization;
        tol = eps(real(eltype(A))),
        debug = false) =
    LinearAlgebra.Eigen(eigQL!(A.diagonals, vectors = Array(A.Q), tol = tol, debug = debug)...)

_eigen!(A::SymTridiagonal;
        tol = eps(real(eltype(A))),
        debug = false) =
    LinearAlgebra.Eigen(eigQL!(A, vectors = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)), tol = tol, debug = debug)...)

_eigen!(A::Hermitian;
        tol = eps(real(eltype(A))),
        debug = false) = _eigen!(symtri!(A), tol = tol, debug = debug)


LinearAlgebra.eigen!(A::SymmetricTridiagonalFactorization;
                     tol = eps(real(eltype(A))),
                     debug = false) = _eigen!(A, tol = tol, debug = debug)

LinearAlgebra.eigen!(A::SymTridiagonal;
                     tol = eps(real(eltype(A))),
                     debug = false) = _eigen!(A, tol = tol, debug = debug)

LinearAlgebra.eigen!(A::Hermitian;
                     tol = eps(real(eltype(A))),
                     debug = false) = _eigen!(A, tol = tol, debug = debug)


function eigen2!(A::SymmetricTridiagonalFactorization;
                 tol = eps(real(float(one(eltype(A))))),
                 debug = false)
    V = zeros(eltype(A), 2, size(A, 1))
    V[1] = 1
    V[end] = 1
    eigQL!(A.diagonals, vectors = rmul!(V, A.Q), tol = tol, debug = debug)
end

function eigen2!(A::SymTridiagonal;
                 tol = eps(real(float(one(eltype(A))))),
                 debug = false)
    V = zeros(eltype(A), 2, size(A, 1))
    V[1] = 1
    V[end] = 1
    eigQL!(A, vectors = V, tol = tol, debug = debug)
end

eigen2!(A::Hermitian;
        tol = eps(float(real(one(eltype(A))))),
        debug = false) = eigen2!(symtri!(A), tol = tol, debug = debug)


eigen2(A::SymTridiagonal;
       tol = eps(float(real(one(eltype(A))))),
       debug = false) = eigen2!(copy(A), tol = tol, debug = debug)

eigen2(A::Hermitian     , tol = eps(float(real(one(eltype(A))))), debug = false) = eigen2!(copy(A), tol = tol, debug = debug)

# First method of each type here is identical to the method defined in
# LinearAlgebra but is needed for disambiguation
function LinearAlgebra.eigvals(A::Hermitian{<:Real})
    T = typeof(sqrt(zero(eltype(A))))
    return eigvals!(LinearAlgebra.copy_oftype(A, T))
end
function LinearAlgebra.eigvals(A::Hermitian{<:Complex})
    T = typeof(sqrt(zero(eltype(A))))
    return eigvals!(LinearAlgebra.copy_oftype(A, T))
end
function LinearAlgebra.eigvals(A::Hermitian)
    T = typeof(sqrt(zero(eltype(A))))
    return eigvals!(LinearAlgebra.copy_oftype(A, T))
end
function LinearAlgebra.eigen(A::Hermitian{<:Real})
    T = typeof(sqrt(zero(eltype(A))))
    return eigen!(LinearAlgebra.copy_oftype(A, T))
end
function LinearAlgebra.eigen(A::Hermitian{<:Complex})
    T = typeof(sqrt(zero(eltype(A))))
    return eigen!(LinearAlgebra.copy_oftype(A, T))
end
function LinearAlgebra.eigen(A::Hermitian)
    T = typeof(sqrt(zero(eltype(A))))
    return eigen!(LinearAlgebra.copy_oftype(A, T))
end

# Aux (should go somewhere else at some point)
function LinearAlgebra.givensAlgorithm(f::Real, g::Real)
    h = hypot(f, g)
    return f/h, g/h, h
end
