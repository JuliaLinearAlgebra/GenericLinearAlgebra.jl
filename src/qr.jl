module QRModule

    using ..HouseholderModule: Householder, HouseholderBlock
    using Base.LinAlg: QR, axpy!

    import Base: getindex, size
    import Base.LinAlg: reflectorApply!

    # @inline function reflectorApply!(A::StridedMatrix, x::AbstractVector, τ::Number) # apply reflector from right. It is assumed that the reflector is calculated on the transposed matrix so we apply Q^T, i.e. no conjugation.
    #     m, n = size(A)
    #     if length(x) != n
    #         throw(DimensionMismatch("reflector must have same length as second dimension of matrix"))
    #     end
    #     @inbounds begin
    #         for i = 1:m
    #             Aiv = A[i, 1]
    #             for j = 2:n
    #                 Aiv += A[i, j]*x[j]'
    #             end
    #             Aiv = Aiv*τ
    #             A[i, 1] -= Aiv
    #             for j = 2:n
    #                 A[i, j] -= Aiv*x[j]
    #             end
    #         end
    #     end
    #     return A
    # end

    @inline function reflectorApply!(A::StridedMatrix, x::AbstractVector, τ::Number) # apply conjugate transpose reflector from right.
        m, n = size(A)
        if length(x) != n
            throw(DimensionMismatch("reflector must have same length as second dimension of matrix"))
        end
        @inbounds begin
            for i = 1:m
                Aiv = A[i, 1]
                for j = 2:n
                    Aiv += A[i, j]*x[j]
                end
                Aiv = Aiv*τ
                A[i, 1] -= Aiv
                for j = 2:n
                    A[i, j] -= Aiv*x[j]'
                end
            end
        end
        return A
    end

    immutable Q{T,S<:QR} <: AbstractMatrix{T}
        data::S
    end

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
        qrUnblocked!(A::StridedMatrix) = LinAlg.qrfactUnblocked!(A)
    end
    function qrBlocked!(A::StridedMatrix, blocksize::Integer, work = Array(eltype(A), blocksize, size(A, 2)))
        m, n = size(A)
        A1 = sub(A, :, 1:min(n, blocksize))
        F = qrUnblocked!(A1)
        if n > blocksize
            A2 = sub(A, :, blocksize + 1:n)
            Ac_mul_B!(F[Tuple{:QBlocked}], A2, sub(work, 1:blocksize, 1:n - blocksize))
            qrBlocked!(sub(A, blocksize + 1:m, blocksize + 1:n), blocksize, work)
        end
        A
    end

    function qrTiledUnblocked!{T,S<:StridedMatrix}(A::UpperTriangular{T,S}, B::StridedMatrix)
        m, n = size(B)
        Ad = A.data
        for i = 1:m
            H = householder!(sub(Ad,i,i), sub(B, :, i))
            Ac_mul_B!(H, )
        end
    end
end