module QRModule

    using ..HouseholderModule: Householder, HouseholderBlock
    using LinearAlgebra: QR, axpy!, reflector!

    import Base: getindex, size
    import LinearAlgebra: reflectorApply!
    import LinearAlgebra

    struct QR2{T,S<:AbstractMatrix{T},V<:AbstractVector{T}} <: Factorization{T}
        factors::S
        τ::V
        QR2{T,S,V}(factors::AbstractMatrix{T}, τ::AbstractVector{T}) where {T,S<:AbstractMatrix,V<:AbstractVector} = 
            new(factors, τ)
    end
    QR2(factors::AbstractMatrix{T}, τ::Vector{T}) where {T} = QR2{T,typeof(factors)}(factors, τ)

    size(F::QR2, i::Integer...) = size(F.factors, i...)

    # Similar to the definition in base but applies the reflector from the right
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

    # FixMe! Consider how to represent Q

    # immutable Q{T,S<:QR2} <: AbstractMatrix{T}
    #     data::S
    # end

    # size(A::Q) = size(A.data)
    # size(A::Q, i::Integer) = size(A.data, i)

    # getindex{T}(A::QR2{T}, ::Type{Tuple{:Q}}) = Q{T,typeof(A)}(A)

    function getindex{T}(A::QR2{T}, ::Type{Tuple{:R}})
        m, n = size(A)
        if m >= n
            UpperTriangular(view(A.factors, 1:n, 1:n))
        else
            throw(ArgumentError("R matrix is trapezoid and cannot be extracted with indexing"))
        end
    end

    function getindex{T}(A::QR2{T}, ::Type{Tuple{:QBlocked}})
        m, n = size(A)
        mmn = min(m,n)
        F = A.factors
        τ = A.τ
        Tmat = zeros(T, mmn, mmn)
        for j = 1:mmn
            for i = 1:j - 1
                Tmat[i,j] = τ[i]*(F[j,i] + dot(view(F, j + 1:m, i), view(F, j + 1:m, j)))
            end
        end
        LinearAlgebra.inv!(LinearAlgebra.UnitUpperTriangular(Tmat))
        for j = 1:min(m,n)
            Tmat[j,j] = τ[j]
            for i = 1:j - 1
                Tmat[i,j] = Tmat[i,j]*τ[j]
            end
        end
        HouseholderBlock{T,typeof(F),Matrix{T}}(F, UpperTriangular(Tmat))
    end

    # qrUnblocked!(A::StridedMatrix) = LinearAlgebra.qrfactUnblocked!(A)
    function qrUnblocked!(A::StridedMatrix{T},
                          τ::StridedVector{T} = fill(zero(T), min(size(A)...))) where {T}

        m, n = size(A)

        # Make a vector view of the first column
        a = view(A, :, 1)
        # Construct the elementary reflector of the first column in-place
        τ1 = reflector!(a)
        # Store reflector loading in τ
        τ[1] = τ1

        # Apply the reflector to the trailing columns if any
        if n > 1
            reflectorApply!(a, τ1, view(A, :, 2:n))
        end

        # Call the function recursively on the submatrix excluding first row and column or return
        if m > 1 && n > 1
            qrUnblocked!(view(A, 2:m, 2:n), view(τ, 2:min(m, n)))
        end

        return QR2{T,typeof(A),typeof(τ)}(A, τ)
    end

    function qrBlocked!(A::StridedMatrix{T},
                        blocksize::Integer = 12,
                        τ::StridedVector{T} = fill(zero(T), min(size(A)...)),
                        work = Matrix{T}(blocksize, size(A, 2))) where T

        m, n = size(A)

        # Make a vector view of the first column block
        A1 = view(A, :, 1:min(n, blocksize))
        # Construct QR factorization of first column block in-place
        F = qrUnblocked!(A1, view(τ, 1:min(m, n, blocksize)))

        # Apply the QR factorization to the trailing columns (if any) and recurse
        if n > blocksize
            # Make a view of the trailing columns
            A2 = view(A, :, blocksize + 1:n)
            # Apply Q' to the trailing columns
            Ac_mul_B!(F[Tuple{:QBlocked}], A2, view(work, 1:min(m, blocksize), 1:(n - blocksize)))
        end

        # Compute QR factorization of trailing block
        if m > blocksize && n > blocksize
            qrBlocked!(view(A, (blocksize + 1):m, (blocksize + 1):n),
                       blocksize,
                       view(τ, (blocksize + 1):min(m, n)),
                       work)
        end

        return QR2{T,typeof(A),typeof(τ)}(A, τ)
    end
end
