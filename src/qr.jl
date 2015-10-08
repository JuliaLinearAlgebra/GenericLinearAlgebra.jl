module QRModule

    using ..HouseholderModule: Householder, HouseholderBlock
    using Base.LinAlg: QR, axpy!

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

    # immutable QR2{T,S<:AbstractMatrix,U} <: Factorization{T}
    #     data::S
    #     reflectors::Vector{U}
    # end

    immutable Q{T,S<:QR} <: AbstractMatrix{T}
        data::S
    end

    # This one is similar to the definition in base
    # function qrUnblocked!{T}(A::StridedMatrix{T})
    #     m, n = size(A)
    #     minmn = min(m,n)
    #     τ = Array(T, minmn)
    #     for i = 1:min(m,n)
    #         x = slice(A, i:m, i)
    #         τi = LinAlg.reflector!(x)
    #         τ[i] = τi
    #         LinAlg.reflectorApply!(x, τi, sub(A, i:m, i + 1:n))
    #     end
    #     QR{T,typeof(A)}(A, τ)
    # end

    # getindex{T,S,U<:Householder}(A::QR2{T,S,U}, ::Type{Val{:Q}}) = Q{T,typeof(A)}(A)
    # function getindex{T,S,U<:Householder}(A::QR2{T,S,U}, ::Type{Val{:R}})
    #     m, n = size(A)
    #     m >= n ? UpperTriangular(sub(A.data, 1:n, 1:n)) : error("R matrix is trapezoid and cannot be extracted with indexing")
    # end

    # This method extracts the Q part of the factorization in the block form H = I - VTV' where V is a matrix the reflectors and T is an upper triangular matrix
    # function getindex{T,S,U}(A::QR2{T,S,U}, ::Type{Val{:QBlocked}})
    #     m, n = size(A)
    #     D = A.data
    #     ref = A.reflectors
    #     Tmat = Array(T, n, n)
    #     Tmat[1,1] = ref[1].τ
    #     for i = 2:n
    #         Tmat[i,i] = ref[i].τ
    #         t12 = sub(Tmat, 1:i-1, i)
    #         Ac_mul_B!(-one(T), sub(D, i+1:m, 1:i-1), sub(D, i+1:m, i), zero(T), t12)
    #         axpy!(-one(T), sub(D, i, 1:i-1), t12)
    #         A_mul_B!(UpperTriangular(sub(Tmat, 1:i-1, 1:i-1)), t12)
    #         scale!(t12, Tmat[i,i])
    #     end
    #     HouseholderBlock{T,typeof(D),Matrix{T}}(D, UpperTriangular(Tmat))
    # end

    getindex{T,S}(A::QR{T,S}, ::Type{Tuple{:Q}}) = Q{T,typeof(A)}(A)
    function getindex{T,S}(A::QR{T,S}, ::Type{Tuple{:R}})
        m, n = size(A)
        m >= n ? UpperTriangular(sub(A.factors, 1:n, 1:n)) : error("R matrix is trapezoid and cannot be extracted with indexing")
    end

    function getindex{T,S}(A::QR{T,S}, ::Type{Tuple{:QBlocked}})
        m, n = size(A)
        F = A.factors
        τ = A.τ
        Tmat = zeros(T, n, n)
        for j = 1:min(m,n)
            for i = 1:j - 1
                Tmat[i,j] = τ[i]*(F[j,i] + dot(sub(F, j + 1:m, i), sub(F, j + 1:m, j)))
            end
        end
        LinAlg.inv!(LinAlg.UnitUpperTriangular(Tmat))
        for j = 1:min(m,n)
            Tmat[j,j] = τ[j]
            for i = 1:j - 1
                Tmat[i,j] = Tmat[i,j]*τ[j]
            end
        end
        HouseholderBlock{T,typeof(F),Matrix{T}}(F, UpperTriangular(Tmat))
    end

    # size(A::QR2) = size(A.data)
    # size(A::QR2, i::Integer) = size(A.data, i)

    size(A::Q) = size(A.data)
    size(A::Q, i::Integer) = size(A.data, i)

    if VERSION < v"0.5.0-"
        qrUnblocked!{T}(A::StridedMatrix{T}) = invoke(LinAlg.qrfact!, (AbstractArray{T,2}, Union{Type{Val{false}},Type{Val{true}}}), A, Val{false})
    else
        using Base.LinAlg: qrUnblocked!
    end
    function qrBlocked!(A::StridedMatrix, blocksize::Integer, work = Array(eltype(A), blocksize, size(A, 2)))
        m, n = size(A)
        A1 = sub(A, 1:m, 1:min(n, blocksize))
        F = LinAlg.qrUnblocked!(A1)
        if n > blocksize
            A2 = sub(A, 1:m, blocksize + 1:n)
            Ac_mul_B!(F[Tuple{:QBlocked}], A2, sub(work, 1:blocksize, 1:n - blocksize))
            qrBlocked!(sub(A, blocksize + 1:m, blocksize + 1:n), blocksize, work)
        end
        A
    end

    function qrTiledUnblocked!{T,S<:StridedMatrix}(A::UpperTriangular{T,S}, B::StridedMatrix)
        m, n = size(B)
        Ad = A.data
        for i = 1:m
            H = householder!(sub(Ad,i,i), sub(B,1:m,i))
            Ac_mul_B!(H, )
        end
    end
end