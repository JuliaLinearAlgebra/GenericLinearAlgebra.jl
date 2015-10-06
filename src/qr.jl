module QRModule

    using ..HouseholderModule: Householder, HouseholderBlock, householder!
    using Base.LinAlg: QR, axpy!
    using ArrayViews

    import Base: getindex, size
    import Base.LinAlg: reflectorApply!

    @inline function reflectorApply!(A::StridedMatrix, x::AbstractVector, τ::Number) # apply reflector from right (assume conjugation)
        m, n = size(A)
        if length(x) != n
            throw(DimensionMismatch("reflector must have same length as second dimension of matrix"))
        end
        @inbounds begin
            for j = 1:n
                vAj = A[1, j]
                for i = 2:m
                    vAj += x[i]'*A[i, j]
                end
                vAj = τ'*vAj
                A[1, j] -= vAj
                for i = 2:m
                    A[i, j] -= x[i]*vAj
                end
            end
        end
        return A
    end

    immutable QR2{T,S<:AbstractMatrix,U} <: Factorization{T}
        data::S
        reflectors::Vector{U}
    end

    immutable Q{T,S<:QR2} <: AbstractMatrix{T}
        data::S
    end

    function qrUnblocked!{T}(A::DenseMatrix{T})
        m, n = size(A)
        minmn = min(m,n)
        τ = Array(Householder{T}, minmn)
        for i = 1:min(m,n)
            H = householder!(view(A,i,i), view(A, i+1:m, i))
            τ[i] = H
            Ac_mul_B!(H, view(A, i:m, i + 1:n))
        end
        QR2{T,typeof(A),eltype(τ)}(A, τ)
    end

    getindex{T,S,U<:Householder}(A::QR2{T,S,U}, ::Type{Val{:Q}}) = Q{T,typeof(A)}(A)
    function getindex{T,S,U<:Householder}(A::QR2{T,S,U}, ::Type{Val{:R}})
        m, n = size(A)
        m >= n ? UpperTriangular(view(A.data, 1:n, 1:n)) : error("R matrix is trapezoid and cannot be extracted with indexing")
    end

    # This method extracts the Q part of the factorization in the block form H = I - VTV' where V is a matrix the reflectors and T is an upper triangular matrix
    function getindex{T,S,U}(A::QR2{T,S,U}, ::Type{Val{:QBlocked}})
        m, n = size(A)
        D = A.data
        ref = A.reflectors
        Tmat = Array(T, n, n)
        Tmat[1,1] = ref[1].τ
        for i = 2:n
            Tmat[i,i] = ref[i].τ
            t12 = view(Tmat, 1:i-1, i)
            Ac_mul_B!(-one(T), view(D, i+1:m, 1:i-1), view(D, i+1:m, i), zero(T), t12)
            axpy!(-one(T), view(D, i, 1:i-1), t12)
            A_mul_B!(UpperTriangular(view(Tmat, 1:i-1, 1:i-1)), t12)
            scale!(t12, Tmat[i,i])
        end
        HouseholderBlock{T,typeof(D),Matrix{T}}(D, UpperTriangular(Tmat))
    end

    size(A::QR2) = size(A.data)
    size(A::QR2, i::Integer) = size(A.data, i)

    size(A::Q) = size(A.data)
    size(A::Q, i::Integer) = size(A.data, i)

    function qrBlocked!(A::DenseMatrix, blocksize::Integer, work = Array(eltype(A), blocksize, size(A, 2)))
        m, n = size(A)
        A1 = view(A, 1:m, 1:min(n, blocksize))
        F = qrUnblocked!(A1)
        if n > blocksize
            A2 = view(A, 1:m, blocksize + 1:n)
            Ac_mul_B!(F[Val{:QBlocked}], A2, view(work, 1:blocksize, 1:n - blocksize))
            qrBlocked!(view(A, blocksize + 1:m, blocksize + 1:n), blocksize, work)
        end
        A
    end

    function qrTiledUnblocked!{T,S<:DenseMatrix}(A::UpperTriangular{T,S}, B::DenseMatrix)
        m, n = size(B)
        Ad = A.data
        for i = 1:m
            H = householder!(view(Ad,i,i), view(B,1:m,i))
            Ac_mul_B!(H, )
        end
    end
end