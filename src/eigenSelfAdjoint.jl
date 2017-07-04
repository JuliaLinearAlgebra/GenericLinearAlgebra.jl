module EigenSelfAdjoint

    using Base.LinAlg: givensAlgorithm

    immutable SymmetricTridiagonalFactorization{T} <: Factorization{T}
        factors::Matrix{T}
        τ::Vector{T}
        diagonals::SymTridiagonal
    end

    Base.size(S::SymmetricTridiagonalFactorization, i::Integer) = size(S.factors, i)

    immutable EigenQ{T} <: AbstractMatrix{T}
        factors::Matrix{T}
        τ::Vector{T}
    end

    getq(S::SymmetricTridiagonalFactorization) = EigenQ(S.factors, S.τ)

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

    function Base.LinAlg.A_mul_B!(Q::EigenQ, B::StridedVecOrMat)
        m, n = size(B)
        if size(Q, 1) != m
            throw(DimensionMismatch(""))
        end
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
        return B
    end

    function Base.LinAlg.A_mul_B!(A::StridedMatrix, Q::EigenQ)
        m, n = size(A)
        if size(Q, 1) != n
            throw(DimensionMismatch(""))
        end
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
        return A
    end

    Base.LinAlg.full(Q::EigenQ) = A_mul_B!(Q, eye(eltype(Q), size(Q, 1), size(Q, 1)))

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

    function eigvalsPWK!{T<:Real}(S::SymTridiagonal{T}, tol = eps(T), debug::Bool=false)
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
                for blockend = blockstart + 1:n
                    if abs(e[blockend - 1]) < tol*sqrt(abs(d[blockend - 1]))*sqrt(abs(d[blockend]))
                        blockend -= 1
                        break
                    end
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
                if blockstart == n
                    break
                end
            end
        end
        sort!(d)
    end

    function eigQL!{T<:Real}(S::SymTridiagonal{T},
                             vectors::Matrix = zeros(T, 0, size(S, 1)),
                             tol = eps(T),
                             debug::Bool=false)
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
                for blockend = blockstart+1:n
                    if abs(e[blockend-1]) < tol*sqrt(abs(d[blockend-1]))*sqrt(abs(d[blockend]))
                        blockend -= 1
                        break
                    end
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
                if blockstart == n
                    break
                end
            end
        end
        p = sortperm(d)
        return d[p], vectors[:,p]
    end

    function eigQR!{T<:Real}(S::SymTridiagonal{T},
                             vectors::Matrix = zeros(T, 0, size(S, 1)),
                             tol = eps(T),
                             debug::Bool=false)
        d = S.dv
        e = S.ev
        n = length(d)
        blockstart = 1
        blockend = n
        @inbounds begin
            while true
                # Check for zero off diagonal elements
                for blockend in (blockstart + 1):n
                    if abs(e[blockend - 1]) <= tol*sqrt(abs(d[blockend - 1]))*sqrt(abs(d[blockend]))
                        blockend -= 1
                        break
                    end
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
                if blockstart == n
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

    function zeroshiftQR!{T}(A::Bidiagonal{T})
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

    # immutable BidiagonalFactorization{T,A<:AbstractMatrix{T}} <: Factorization{T}
        # matrix::A
        # rotationLeft::Rotation{T}
        # rotationRigth::Rotation{T}
    # end

    symtri!(A::Hermitian) = A.uplo == 'L' ? symtriLower!(A.data) : symtriUpper!(A.data)
    symtri!{T<:Real}(A::Symmetric{T}) = A.uplo == 'L' ? symtriLower!(A.data) : symtriUpper!(A.data)

    function symtriLower!{T}(AS::Matrix{T}) # Assume that lower triangle stores the relevant part
        n = size(AS,1)
        τ = zeros(T,n-1)
        u = Matrix{T}(n, 1)
        @inbounds begin
        for k = 1:(n - 2 + !(T<:Real))
            τk = LinAlg.reflector!(view(AS, k + 1:n, k))
            τ[k] = τk

            for i = k+1:n
                u[i] = AS[i,k+1]
            end
            for j = k+2:n
                ASjk = AS[j,k]
                for i = j:n
                    u[i] += AS[i,j]*ASjk
                end
            end
            for j = k+1:n-1
                tmp = zero(T)
                for i = j+1:n
                    tmp += AS[i,j]'AS[i,k]
                end
                u[j] += tmp
            end

            vcAv = u[k+1]
            for i = k+2:n
                vcAv += AS[i,k]'u[i]
            end
            ξτ2 = real(vcAv)*abs2(τk)/2

            u[k+1] = u[k+1]*τk - ξτ2
            for i = k+2:n
                u[i] = u[i]*τk - ξτ2*AS[i,k]
            end

            AS[k+1,k+1] -= 2real(u[k+1])
            for i = k+2:n
                AS[i,k+1] -= u[i] + AS[i,k]*u[k+1]'
            end
            for j = k+2:n
                ASjk = AS[j,k]
                uj = u[j]
                AS[j,j] -= 2real(uj*ASjk')
                for i = j+1:n
                    AS[i,j] -= u[i]*ASjk' + AS[i,k]*uj'
                end
            end
        end
        end
        SymmetricTridiagonalFactorization(AS,τ,SymTridiagonal(real(diag(AS)),real(diag(AS,-1))))
    end

    eigvals!(         A::SymmetricTridiagonalFactorization, tol = eps(real(float(one(eltype(A))))), debug = false) = eigvalsPWK!(A.diagonals, tol, debug)
    eigvals!(         A::SymTridiagonal,                    tol = eps(real(float(one(eltype(A))))), debug = false) = eigvalsPWK!(A, tol, debug)
    eigvals!{T<:Real}(A::LinAlg.RealHermSymComplexHerm{T},  tol = eps(real(float(one(eltype(A))))), debug = false) = eigvals!(symtri!(A), tol, debug)

    eig!(A::SymmetricTridiagonalFactorization, tol = eps(real(float(one(eltype(A))))), debug = false) = eigQL!(A.diagonals, full(getq(A)), tol, debug)
    eig!(A::SymTridiagonal,                    tol = eps(real(float(one(eltype(A))))), debug = false) = eigQL!(A, eye(eltype(A), size(A, 1)), tol, debug)
    eig!(A::LinAlg.RealHermSymComplexHerm,     tol = eps(real(float(one(eltype(A))))), debug = false) = eig!(symtri!(A), tol, debug)

    function eig2!(A::SymmetricTridiagonalFactorization, tol = eps(real(float(one(eltype(A))))), debug = false)
        V = zeros(eltype(A), 2, size(A, 1))
        V[1] = 1
        V[end] = 1
        eigQL!(A.diagonals, A_mul_B!(V, getq(A)), tol, debug)
    end
    function eig2!(A::SymTridiagonal, tol = eps(real(float(one(eltype(A))))), debug = false)
        V = zeros(eltype(A), 2, size(A, 1))
        V[1] = 1
        V[end] = 1
        eigQL!(A, V, tol, debug)
    end
    eig2!(A::LinAlg.RealHermSymComplexHerm,   tol = eps(real(float(one(eltype(A))))), debug = false) = eig2!(symtri!(A), tol, debug)

    eigvals(A::SymTridiagonal,                tol = eps(real(float(one(eltype(A))))), debug = false) = eigvals!(copy(A), tol, debug)
    eigvals(A::LinAlg.RealHermSymComplexHerm, tol = eps(real(float(one(eltype(A))))), debug = false) = eigvals!(copy(A), tol, debug)

    eig(A::SymTridiagonal               ,     tol = eps(real(float(one(eltype(A))))), debug = false) = eig!(copy(A)    , tol, debug)
    eig(A::LinAlg.RealHermSymComplexHerm,     tol = eps(real(float(one(eltype(A))))), debug = false) = eig!(copy(A)    , tol, debug)

    eig2(A::SymTridiagonal               ,    tol = eps(real(float(one(eltype(A))))), debug = false) = eig2!(copy(A)   , tol, debug)
    eig2(A::LinAlg.RealHermSymComplexHerm,    tol = eps(real(float(one(eltype(A))))), debug = false) = eig2!(copy(A)   , tol, debug)

    # Aux (should go somewhere else at some point)
    function Base.LinAlg.givensAlgorithm(f::Real, g::Real)
        h = hypot(f, g)
        return f/h, g/h, h
    end

end