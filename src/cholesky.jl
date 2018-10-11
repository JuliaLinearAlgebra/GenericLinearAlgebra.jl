using LinearAlgebra: rdiv!

function cholUnblocked!(A::AbstractMatrix{T}, ::Type{Val{:L}}) where T<:Number
    n = LinearAlgebra.checksquare(A)
    A[1,1] = sqrt(A[1,1])
    if n > 1
        a21 = view(A, 2:n, 1)
        rmul!(a21, inv(real(A[1,1])))

        A22 = view(A, 2:n, 2:n)
        rankUpdate!(Hermitian(A22, :L), a21, -1)
        cholUnblocked!(A22, Val{:L})
    end
    A
end

function cholBlocked!(A::AbstractMatrix{T}, ::Type{Val{:L}}, blocksize::Integer) where T<:Number
    n = LinearAlgebra.checksquare(A)
    mnb = min(n, blocksize)
    A11 = view(A, 1:mnb, 1:mnb)
    cholUnblocked!(A11, Val{:L})
    if n > blocksize
        A21 = view(A, (blocksize + 1):n, 1:blocksize)
        rdiv!(A21, LowerTriangular(A11)')

        A22 = view(A, (blocksize + 1):n, (blocksize + 1):n)
        rankUpdate!(Hermitian(A22, :L), A21, -1)
        cholBlocked!(A22, Val{:L}, blocksize)
    end
    A
end

function cholRecursive!(A::StridedMatrix{T}, ::Type{Val{:L}}, cutoff = 1) where T
    n = LinearAlgebra.checksquare(A)
    if n == 1
        A[1,1] = sqrt(A[1,1])
    elseif n < cutoff
        cholUnblocked!(A, Val{:L})
    else
        n2 = div(n, 2)
        A11 = view(A, 1:n2, 1:n2)
        cholRecursive!(A11, Val{:L})
        A21 = view(A, n2 + 1:n, 1:n2)
        rdiv!(A21, LowerTriangular(A11)')

        A22 = view(A, n2 + 1:n, n2 + 1:n)
        rankUpdate!(Hermitian(A22, :L), A21, -1)
        cholRecursive!(A22, Val{:L}, cutoff)
    end
    return LowerTriangular(A)
end
