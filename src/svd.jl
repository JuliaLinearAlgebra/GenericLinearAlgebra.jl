using LinearAlgebra

import LinearAlgebra: mul!

mul!(G::LinearAlgebra.Givens, ::Nothing) = nothing
mul!(::Nothing, G::LinearAlgebra.Givens) = nothing

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

function svdIter!(B::Bidiagonal{T}, n1, n2, shift, U = nothing, Vt = nothing) where T<:Real

    if istriu(B)

        d = B.dv
        e = B.ev

        G, r = givens(d[n1] - abs2(shift)/d[n1], e[n1], 1, 2)
        mul!(G, Vt)

        ditmp       = d[n1]
        ei1         = e[n1]
        di          = ditmp*G.c + ei1*G.s
        ei1         = -ditmp*G.s + ei1*G.c
        di1         = d[n1 + 1]
        bulge       = di1*G.s
        di1        *= G.c

        for i = n1:n2 - 2
            G, r      = givens(di, bulge, i, i + 1)
            mul!(U, G')
            d[i]      = G.c*di + G.s*bulge
            ei        = G.c*ei1 + G.s*di1
            di1       = -G.s*ei1 + G.c*di1
            ei1       = e[i + 1]
            bulge     = G.s*ei1
            ei1      *= G.c

            G, r      = givens(ei, bulge, i + 1, i + 2)
            mul!(G, Vt)
            e[i]      = ei*G.c + bulge*G.s
            di        = di1*G.c + ei1*G.s
            ei1       = -di1*G.s + ei1*G.c
            di2       = d[i + 2]
            bulge     = di2*G.s
            di1       = di2*G.c
        end

        G, r      = givens(di, bulge, n2 - 1, n2)
        mul!(U, G')
        d[n2 - 1] = G.c*di + G.s*bulge
        e[n2 - 1] = G.c*ei1 + G.s*di1
        d[n2]     = -G.s*ei1 + G.c*di1

    else
        throw(ArgumentError("lower bidiagonal version not implemented yet"))
    end

    return B
end

# See LAWN 3
function svdDemmelKahan!(B::Bidiagonal{T}, n1, n2, U = nothing, Vt = nothing) where T<:Real

    if istriu(B)

        d = B.dv
        e = B.ev

        oldcs = one(T)
        G     = LinearAlgebra.Givens(1, 2, one(T), zero(T))
        Gold  = G

        for i = n1:n2 - 1
            G, r  = givens(d[i] * G.c, e[i], i, i + 1)
            mul!(G, Vt) # FixMe! Vt update might be wrong
            if i != n1
                e[i - 1] = Gold.s * r
            end

            Gold, d[i] = givens(Gold.c * r, d[i + 1] * G.s, i + 1, i + 2)
            mul!(U, G')  # FixMe! U update might be wrong
        end

        h         = d[n2] * G.c
        e[n2 - 1] = h * Gold.s
        d[n2]     = h * Gold.c

    else
        throw(ArgumentError("lower bidiagonal version not implemented yet"))
    end

    return B
end

function _svdvals!(B::Bidiagonal{T}; tol = eps(T), debug = false) where T<:Real

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
                    return sort(abs.(diag(B)), rev = true) # done
                else
                    tolcritTop = tol * abs(d[n2i - 1] * d[n2i])

                    # debug && println("n2i=", n2i, ", d[n2i-1]=", d[n2i-1], ", d[n2i]=", d[n2i],
                        # ", e[n2i-1]=", e[n2i-1], ", tolcritTop=", tolcritTop)

                    if abs(e[n2i - 1]) > tolcritTop
                        n2 = n2i
                        break
                    end
                end
            end

            # Search for largest sub-bidiagonal matrix ending at n2
            for _n1 = (n2 - 1):-1:1

                if _n1 == 1
                    n1 = _n1
                    break
                else
                    tolcritBottom = tol * abs(d[_n1 - 1] * d[_n1])

                    # debug && println("n1=", n1, ", d[n1]=", d[n1], ", d[n1-1]=", d[n1-1], ", e[n1-1]", e[n1-1],
                        # ", tolcritBottom=", tolcritBottom)

                    if abs(e[_n1 - 1]) < tolcritBottom
                        n1 = _n1
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

            # FixMe! Calling the zero shifted version only the first time is adhoc but seems
            # to work well. Reexamine analysis in LAWN 3 to improve this later.
            if count == 0
                svdDemmelKahan!(B, n1, n2)
            else
                shift = svdvals2x2(d[n2 - 1], d[n2], e[n2 - 1])[1]
                svdIter!(B, n1, n2, shift)
            end
            count += 1
            debug && println("count=", count)
        end
    else
        # Just transpose the matrix. Since we are only interested in the
        # values here it doesn't matter.
        return svdvals!(Bidiagonal(d, e, :U); tol = tol, debug = debug)
    end
end

### FixMe!!! svd! doesn't update vectors in all branches. Comment out 7 April 2016

# function svd!{T<:Real}(B::Bidiagonal{T}, tol = eps(T))
#     n = size(B, 1)
#     n2 = n
#     d = B.dv
#     e = B.ev
#     U  = Matrix{T}(I, n, n)
#     Vt = Matrix{T}(I, n, n)
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

function bidiagonalize!(A::AbstractMatrix)
    m, n = size(A)
    τl, τr = eltype(A)[], eltype(A)[]

    if m >= n
        # tall case: lower bidiagonal
        for i = 1:min(m, n)
            x = view(A, i:m, i)
            τi = LinearAlgebra.reflector!(x)
            push!(τl, τi)
            LinearAlgebra.reflectorApply!(x, τi, view(A, i:m, i + 1:n))
            if i < n
                x = view(A, i, i + 1:n)
                conj!(x)
                τi = LinearAlgebra.reflector!(x)
                push!(τr, τi)
                LinearAlgebra.reflectorApply!(view(A, i + 1:m, i + 1:n), x, τi)
            end
        end
        return Bidiagonal(real(diag(A)), real(diag(A, 1)), :U), A, τl, τr
    else
        # wide case: upper bidiagonal
        for i = 1:min(m, n)
            x = view(A, i, i:n)
            τi = LinearAlgebra.reflector!(x)
            push!(τr, τi)
            LinearAlgebra.reflectorApply!(view(A, i + 1:m, i:n), x, τi)
            if i < m
                x = view(A, i + 1:m, i)
                τi = LinearAlgebra.reflector!(x)
                push!(τl, τi)
                LinearAlgebra.reflectorApply!(x, τi, view(A, i + 1:m, i + 1:n))
            end
        end
        return Bidiagonal(real(diag(A)), real(diag(A, -1)), :L), A, τl, τr
    end
end

_svdvals!(A::StridedMatrix; tol = eps(T), debug = false) = svdvals!(bidiagonalize!(A)[1], tol = tol, debug = debug)

LinearAlgebra.svdvals!(B::Bidiagonal{T}; tol = eps(T), debug = false) where T<:Real = _svdvals!(B, tol = tol, debug = debug)
LinearAlgebra.svdvals!(A::StridedMatrix; tol = eps(real(eltype(A))), debug = false) = _svdvals!(A, tol = tol, debug = debug)
