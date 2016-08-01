# Eigenfunctionality

module EigenGeneral

    using ..HouseholderModule: Householder
    using Base.LinAlg: Givens, Rotation

    import Base: A_mul_B!, A_mul_Bc!, Ac_mul_B, A_mul_Bc, A_ldiv_B!, ctranspose, full, getindex, size
    import Base.LinAlg: QRPackedQ

    using Compat
    import Compat.view

    # Auxiliary
    function adiagmax(A::StridedMatrix)
        adm = zero(typeof(real(A[1])))
        @inbounds begin
            for i = size(A,1)
                adm = max(adm, abs(A[i,i]))
            end
        end
        return adm
    end

    # Hessenberg Matrix
    immutable HessenbergMatrix{T,S<:StridedMatrix} <: AbstractMatrix{T}
        data::S
    end

    copy{T,S}(H::HessenbergMatrix{T,S}) = HessenbergMatrix{T,S}(copy(H.data))

    getindex{T,S}(H::HessenbergMatrix{T,S}, i::Integer, j::Integer) = i > j + 1 ? zero(T) : H.data[i,j]

    size(H::HessenbergMatrix) = size(H.data)
    size(H::HessenbergMatrix, i::Integer) = size(H.data, i)

    function A_ldiv_B!(H::HessenbergMatrix, B::AbstractVecOrMat)
        n = size(H, 1)
        Hd = H.data
        for i = 1:n-1
            G, _ = givens!(Hd, i, i+1, i)
            A_mul_B!(G, view(Hd, 1:n, i+1:n))
            A_mul_B!(G, B)
        end
        A_ldiv_B!(Triangular(Hd, :U), B)
    end

    # Hessenberg factorization
    immutable HessenbergFactorization{T, S<:StridedMatrix,U} <: Factorization{T}
        data::S
        τ::Vector{U}
    end

    function hessfact!{T}(A::StridedMatrix{T})
        n = Compat.LinAlg.checksquare(A)
        τ = Array(Householder{T}, n - 1)
        for i = 1:n - 1
            xi = view(A, i + 1:n, i)
            t  = LinAlg.reflector!(xi)
            H  = Householder{T,typeof(xi)}(view(xi, 2:n - i), t)
            τ[i] = H
            Ac_mul_B!(H, view(A, i + 1:n, i + 1:n))
            A_mul_B!(view(A, :, i + 1:n), H)
        end
        return HessenbergFactorization{T, typeof(A), eltype(τ)}(A, τ)
    end

    size(H::HessenbergFactorization, args...) = size(H.data, args...)

    # Schur
    immutable Schur{T,S<:StridedMatrix} <: Factorization{T}
        data::S
        R::Rotation
    end

    function schurfact!{T<:Real}(H::HessenbergFactorization{T}; tol = eps(T), debug = false)
        n = size(H, 1)
        istart = 1
        iend = n
        HH = H.data
        τ = Rotation(Givens{T}[])
        @inbounds begin
        while true
            # Determine if the matrix splits. Find lowest positioned subdiagonal "zero"
            for istart = iend - 1:-1:1
                # debug && @printf("istart: %6d, iend %6d\n", istart, iend)
                # istart == minstart && break
                if abs(HH[istart + 1, istart]) < tol*(abs(HH[istart, istart]) + abs(HH[istart + 1, istart + 1]))
                    istart += 1
                    debug && @printf("Top deflation! Subdiagonal element is: %10.3e and istart now %6d\n", HH[istart, istart - 1], istart)
                    break
                elseif istart > 1 && abs(HH[istart, istart - 1]) < tol*(abs(HH[istart - 1, istart - 1]) + abs(HH[istart, istart]))
                    debug && @printf("Top deflation! Next subdiagonal element is: %10.3e and istart now %6d\n", HH[istart, istart - 1], istart)
                    break
                end
            end

            # if block size is one we deflate
            if istart >= iend
                iend -= 1

            # and the same for a 2x2 block
            elseif istart + 1 == iend
                iend -= 2

            # if we don't deflate we'll run either a single or double shift bulge chase
            else
                Hmm = HH[iend, iend]
                Hm1m1 = HH[iend - 1, iend - 1]
                d = Hm1m1*Hmm - HH[iend, iend - 1]*HH[iend - 1, iend]
                t = Hm1m1 + Hmm
                debug && @printf("block start is: %6d, block end is: %6d, d: %10.3e, t: %10.3e\n", istart, iend, d, t)

                # For small (sub) problems use Raleigh quotion shift and single shift
                if iend <= istart + 2
                    σ = HH[iend, iend]

                    # Run a bulge chase
                    singleShiftQR!(HH, τ, σ, istart, iend)

                # If the eigenvales of the 2x2 block are real use single shift
                elseif t*t > 4d
                    debug && @printf("Single shift! subdiagonal is: %10.3e\n", HH[iend, iend - 1])

                    # Calculate the Wilkinson shift
                    λ1 = 0.5*(t + sqrt(t*t - 4d))
                    λ2 = 0.5*(t - sqrt(t*t - 4d))
                    σ = abs(Hmm - λ1) < abs(Hmm - λ2) ? λ1 : λ2

                    # Run a bulge chase
                    singleShiftQR!(HH, τ, σ, istart, iend)

                # else use double shift
                else
                    debug && @printf("Double shift! subdiagonal is: %10.3e, last subdiagonal is: %10.3e\n", HH[iend, iend - 1], HH[iend - 1, iend - 2])
                    doubleShiftQR!(HH, τ, t, d, istart, iend)
                end
            end
            if iend <= 2 break end
        end
        end
        return Schur{T,typeof(HH)}(HH, τ)
    end
    schurfact!(A::StridedMatrix; tol = eps(), debug = false) = schurfact!(hessfact!(A), tol = tol, debug = debug)

    function singleShiftQR!(HH::StridedMatrix, τ::Rotation, shift::Number, istart::Integer, iend::Integer)
        m = size(HH, 1)
        H11 = HH[istart, istart]
        H21 = HH[istart + 1, istart]
        if m > istart + 1
            Htmp = HH[istart + 2, istart]
            HH[istart + 2, istart] = 0
        end
        G, _ = givens(H11 - shift, H21, istart, istart + 1)
        A_mul_B!(G, view(HH, :, istart:m))
        A_mul_Bc!(view(HH, 1:min(istart + 2, iend), :), G)
        A_mul_B!(G, τ)
        for i = istart:iend - 2
            G, _ = givens(HH[i + 1, i], HH[i + 2, i], i + 1, i + 2)
            A_mul_B!(G, view(HH, :, i:m))
            HH[i + 2, i] = Htmp
            if i < iend - 2
                Htmp = HH[i + 3, i + 1]
                HH[i + 3, i + 1] = 0
            end
            A_mul_Bc!(view(HH, 1:min(i + 3, iend), :), G)
            # A_mul_B!(G, τ)
        end
        return HH
    end

    function doubleShiftQR!(HH::StridedMatrix, τ::Rotation, shiftTrace::Number, shiftDeterminant::Number, istart::Integer, iend::Integer)
        m = size(HH, 1)
        H11 = HH[istart, istart]
        H21 = HH[istart + 1, istart]
        Htmp11 = HH[istart + 2, istart]
        HH[istart + 2, istart] = 0
        Htmp21 = HH[istart + 3, istart]
        HH[istart + 3, istart] = 0
        Htmp22 = HH[istart + 3, istart + 1]
        HH[istart + 3, istart + 1] = 0
        G1, r = givens(H11*H11 + HH[istart, istart + 1]*H21 - shiftTrace*H11 + shiftDeterminant, H21*(H11 + HH[istart + 1, istart + 1] - shiftTrace), istart, istart + 1)
        G2, _ = givens(r, H21*HH[istart + 2, istart + 1], istart, istart + 2)
        vHH = view(HH, :, istart:m)
        A_mul_B!(G1, vHH)
        A_mul_B!(G2, vHH)
        vHH = view(HH, 1:istart + 3, :)
        A_mul_Bc!(vHH, G1)
        A_mul_Bc!(vHH, G2)
        A_mul_B!(G1, τ)
        A_mul_B!(G2, τ)
        for i = istart:iend - 2
            for j = 1:2
                if i + j + 1 > iend break end
                # G, _ = givens(H.H,i+1,i+j+1,i)
                G, _ = givens(HH[i + 1, i], HH[i + j + 1, i], i + 1, i + j + 1)
                A_mul_B!(G, view(HH, :, i:m))
                HH[i + j + 1, i] = Htmp11
                Htmp11 = Htmp21
                # if i + j + 2 <= iend
                    # Htmp21 = HH[i + j + 2, i + 1]
                    # HH[i + j + 2, i + 1] = 0
                # end
                if i + 4 <= iend
                    Htmp22 = HH[i + 4, i + j]
                    HH[i + 4, i + j] = 0
                end
                A_mul_Bc!(view(HH, 1:min(i + j + 2, iend), :), G)
                # A_mul_B!(G, τ)
            end
        end
        return HH
    end

    eigvals!(A::StridedMatrix; tol = eps(one(A[1])), debug = false) = eigvals!(schurfact!(A, tol = tol, debug = debug))
    eigvals!(H::HessenbergMatrix; tol = eps(one(A[1])), debug = false) = eigvals!(schurfact!(H, tol = tol, debug = debug))
    eigvals!(H::HessenbergFactorization; tol = eps(one(A[1])), debug = false) = eigvals!(schurfact!(H, tol = tol, debug = debug))

    function eigvals!{T}(S::Schur{T}; tol = eps(T))
        HH = S.data
        n = size(HH, 1)
        vals = Array(Complex{T}, n)
        i = 1
        while i < n
            Hii = HH[i, i]
            Hi1i1 = HH[i + 1, i + 1]
            if abs(HH[i + 1, i]) < tol*(abs(Hi1i1) + abs(Hii))
                vals[i] = Hii
                i += 1
            else
                d = Hii*Hi1i1 - HH[i, i + 1]*HH[i + 1, i]
                t = Hii + Hi1i1
                x = 0.5*t
                y = sqrt(complex(x*x - d))
                vals[i] = x + y
                vals[i + 1] = x - y
                i += 2
            end
        end
        if i == n vals[i] = HH[n, n] end
        return vals
    end
end