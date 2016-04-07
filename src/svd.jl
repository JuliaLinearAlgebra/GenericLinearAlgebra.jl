module SVDModule

import Base: A_mul_B!, A_mul_Bc!

A_mul_B!(G::LinAlg.Givens, ::Void) = nothing
A_mul_Bc!(::Void, G::LinAlg.Givens) = nothing

function svdvals2x2(d1, d2, e)
    d1sq = d1*d1
    d2sq = d2*d2
    esq  = e*e
    b = d1sq + d2sq + esq
    D = sqrt(abs2((d1 + d2)*(d1 - d2)) + esq*(2*(d1sq + d2sq) + esq))
    D2 = b + D
    λ1 = 2*d1sq*d2sq/D2
    λ2 = D2/2
    return minmax(sqrt(λ1), sqrt(λ2))
end

function svdIter!{T<:Real}(B::Bidiagonal{T}, n1, n2, shift, U = nothing, Vt = nothing)

    if istriu(B)

        d = B.dv
        e = B.ev

        G, r = givens(d[n1] - abs2(shift)/d[n1], e[n1], 1, 2)
        A_mul_B!(G, Vt)

        ditmp       = d[n1]
        ei1         = e[n1]
        di          = ditmp*G.c + ei1*G.s
        ei1         = -ditmp*G.s + ei1*G.c
        di1         = d[n1 + 1]
        bulge       = di1*G.s
        di1        *= G.c

        for i = n1:n2 - 2
            G, r      = givens(di, bulge, i, i + 1)
            A_mul_Bc!(U, G)
            d[i]      = G.c*di + G.s*bulge
            ei        = G.c*ei1 + G.s*di1
            di1       = -G.s*ei1 + G.c*di1
            ei1       = e[i + 1]
            bulge     = G.s*ei1
            ei1      *= G.c

            G, r      = givens(ei, bulge, i + 1, i + 2)
            A_mul_B!(G, Vt)
            e[i]      = ei*G.c + bulge*G.s
            di        = di1*G.c + ei1*G.s
            ei1       = -di1*G.s + ei1*G.c
            di2       = d[i + 2]
            bulge     = di2*G.s
            di1       = di2*G.c
        end

        G, r      = givens(di, bulge, n2 - 1, n2)
        A_mul_Bc!(U, G)
        d[n2 - 1] = G.c*di + G.s*bulge
        e[n2 - 1] = G.c*ei1 + G.s*di1
        d[n2]     = -G.s*ei1 + G.c*di1

    else
        throw(ArgumentError("lower bidiagonal version not implemented yet"))
    end

    return B
end

function svdvals!{T<:Real}(B::Bidiagonal{T}, tol = eps(T); debug = false)

    n = size(B, 1)
    n1 = 1
    n2 = n
    d = B.dv
    e = B.ev
    count = 0

    if istriu(B)
        while true

            # Search for biggest index for non-zero off diagonal value in e
            for n2i = n2:-1:1
                if n2i == 1
                    return sort(abs(diag(B)), rev = true), count # done
                else
                    # FixMe!. Use + until zero inflated algorithm has been implemented. Then
                    # use * because the precision will be much higher then.
                    tolcritTop = tol * (abs(d[n2i - 1]) + abs(d[n2i]))

                    debug && println("n2i=", i, ", d[n2i-1]=", d[n2i-1], ", d[n2i]=", d[n2i],
                        ", e[n2i-1]=", e[n2i-1], ", tolcritTop=", tolcritTop)

                    if abs(e[n2i - 1]) > tolcritTop
                        n2 = n2i
                        break
                    end
                end
            end

            # Search for largest sub-bidiagonal matrix ending at n2
            for n1 = n2 - 1:-1:1

                if n1 == 1
                    break
                else
                    # FixMe!. Use + until zero inflated algorithm has been implemented. Then
                    # use * because the precision will be much higher then.
                    tolcritBottom = tol * (abs(d[n1 - 1]) + abs(d[n1]))

                    debug && println("n1=", n1, ", d[n1]=", d[n1], ", d[n1-1]=", d[n1-1], ", e[n1-1]", e[n1-1],
                        ", tolcritBottom=", tolcritBottom)

                    if abs(e[n1 - 1]) < tolcritBottom
                        break
                    end
                end
            end

            # 2x2 block
            if n2 - n1 == 1
                s1, s2 = svdvals2x2(d[n1], d[n2], e[n1])
                d[n1] = s2
                d[n2] = s1
                e[n1] = 0
            end

            debug && println("n1=", n1, ", n2=", n2, ", d[n1]=", d[n1], ", d[n2]=", d[n2], ", e[n1]=", e[n1])

            shift = svdvals2x2(d[n2 - 1], d[n2], e[n2 - 1])[1]
            svdIter!(B, n1, n2, ifelse(count == 0, zero(shift), shift))
            count += 1
        end
    else
        throw(ArgumentError("lower bidiagonal version not implemented yet"))
    end
end

### FixMe!!! svd! doesn't update vectors in all branches. Comment out 7 April 2016

# function svd!{T<:Real}(B::Bidiagonal{T}, tol = eps(T))
#     n = size(B, 1)
#     n2 = n
#     d = B.dv
#     e = B.ev
#     U = eye(T, n)
#     Vt = eye(T, n)
#     count = 0

#     if istriu(B)
#         while true
#             while abs(e[n2 - 1]) < tol*abs(d[n2 - 1])*abs(d[n2])
#                 n2 -= 1
#                 if n2 == 1
#                     return sort(abs(diag(B)), rev = true), count # done
#                 end
#             end
#             n1 = n2 - 1
#             while n1 > 1 && abs(e[n1 - 1]) > tol*abs(d[n1 - 1])*abs(d[n1])
#                 n1 -= 1
#             end
#             if n2 - n1 == 1 # 2x2 block
#                 s1, s2 = svdvals2x2(d[n1], d[n2], e[n1])
#                 d[n1] = s2
#                 d[n2] = s1
#                 e[n1] = 0
#             end

#             shift = svdvals2x2(d[n2 - 1], d[n2], e[n2 - 1])[1]
#             svdIter!(B, n1, n2, ifelse(count == 0, zero(shift), shift), U, Vt)
#             count += 1
#         end
#     else
#         throw(ArgumentError("lower bidiagonal version not implemented yet"))
#     end
# end

function elementaryLeftAndApply!(A::AbstractMatrix, row::Integer, column::Integer)
    τ = LinAlg.elementaryLeft!(A, row, column)
    for j = column + 1:size(A, 2)
        tmp = A[row, j]
        for i = row + 1:size(A, 1)
            tmp += A[i, column]'*A[i, j]
        end
        tmp *= τ'
        A[row,j] -= tmp
        for i = row + 1:size(A, 1)
            A[i, j] -= A[i, column]*tmp
        end
    end
    return τ
end

function elementaryRightAndApply!(A::AbstractMatrix, row::Integer, column::Integer)
    τ = LinAlg.elementaryRight!(A, row, column)
    for i = row + 1:size(A, 1)
        tmp = A[i, column]
        for j = column + 1:size(A, 2)
            tmp += A[i, j]*A[row, j]'
        end
        tmp *= τ
        A[i, column] -= tmp
        for j = column + 1:size(A, 2)
            A[i, j] -= tmp*A[row, j]
        end
    end
    return τ
end

function bidiagonalize!(A::AbstractMatrix)
    m, n = size(A)
    τl, τr = eltype(A)[], eltype(A)[]

    if m >= n
        # tall case: lower bidiagonal
        for i = 1:min(m, n)
            x = slice(A, i:m, i)
            τi = LinAlg.reflector!(x)
            push!(τl, τi)
            LinAlg.reflectorApply!(x, τi, slice(A, i:m, i + 1:n))
            if i < n
                x = slice(A, i, i + 1:n)
                conj!(x)
                τi = LinAlg.reflector!(x)
                push!(τr, τi)
                LinAlg.reflectorApply!(slice(A, i + 1:m, i + 1:n), x, τi)
            end
        end
        return Bidiagonal(real(diag(A)), real(diag(A, 1)), true), A, τl, τr
    else
        # wide vase: upper bidiagonal
        for i = 1:min(m, n)
            x = slice(A, i, i:n)
            conj(x)
            τi = LinAlg.reflector!(x)
            push!(τr, τi)
            LinAlg.reflectorApply!(x, τi, slice(A, i + 1:m, i:n))
            if i < m
                x = slice(A, i + 1:m, i)
                τi = LinAlg.reflector!(x)
                push!(τl, τi)
                LinAlg.reflectorApply!(slice(A, i + 1:m, i + 1:n), x, τi)
            end
        end
        return Bidiagonal(real(diag(A)), real(diag(A, -1)), false), A, τl, τr
    end
end

svdvals!(A::StridedMatrix) = svdvals!(bidiagonalize!(A)[1])

end #module