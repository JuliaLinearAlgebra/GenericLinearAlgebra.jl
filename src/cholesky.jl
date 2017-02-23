module CholeskyModule

    using ..JuliaBLAS
    using Base.LinAlg: A_rdiv_Bc!

    function cholUnblocked!{T<:Number}(A::AbstractMatrix{T}, ::Type{Val{:L}})
        n = LinAlg.checksquare(A)
        A[1,1] = sqrt(A[1,1])
        if n > 1
            a21 = view(A, 2:n, 1)
            scale!(a21, inv(real(A[1,1])))

            A22 = view(A, 2:n, 2:n)
            rankUpdate!(-one(T), a21, Hermitian(A22, :L))
            cholUnblocked!(A22, Val{:L})
        end
        A
    end

    function cholBlocked!{T<:Number}(A::AbstractMatrix{T}, ::Type{Val{:L}}, blocksize::Integer)
        n = LinAlg.checksquare(A)
        mnb = min(n, blocksize)
        A11 = view(A, 1:mnb, 1:mnb)
        cholUnblocked!(A11, Val{:L})
        if n > blocksize
            A21 = view(A, blocksize+1:n, 1:blocksize)
            A_rdiv_Bc!(A21, LowerTriangular(A11))

            A22 = view(A, blocksize+1:n, blocksize+1:n)
            rankUpdate!(-real(one(T)), A21, Hermitian(A22, :L))
            cholBlocked!(A22, Val{:L}, blocksize)
        end
        A
    end

    function cholRec!{T}(A::StridedMatrix{T}, ::Type{Val{:L}}, cutoff = 1)
        n = LinAlg.checksquare(A)
        if n == 1
            A[1,1] = sqrt(A[1,1])
        elseif n < cutoff
            cholUnblocked!(A, Val{:L})
        else
            n2 = div(n, 2)
            A11 = view(A, 1:n2, 1:n2)
            cholRec!(A11, Val{:L})
            A21 = view(A, n2 + 1:n, 1:n2)
            A_rdiv_Bc!(A21, LowerTriangular(A11))

            A22 = view(A, n2 + 1:n, n2 + 1:n)
            rankUpdate!(-real(one(T)), A21, Hermitian(A22, :L))
            cholRec!(A22, Val{:L}, cutoff)
        end
        return LowerTriangular(A)
    end

end