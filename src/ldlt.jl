import Base: *

# When the lower triangle contains the input it is actually LDLt
# Only the lower triangle of the input matrix A is referenced
#
# [α   ] = [1 0 ] [δ 0 ] [1 l' ] = [δ         δ*l'        ]
# [a A₋]   [l L₋] [0 D₋] [0 L₋']   [l*δ l*δ*l' + L₋*D₋*L₋']
#
# so
#
# δ = α
# l = δ \ a
#
# followed by the recursion on the updated lower block
#
# L₋'*D₋*L₋' = A₋ - l*δ*l'
function _ldlt_lower!(A::StridedMatrix)
    n = size(A, 2)
    δ = A[1, 1]
    for i in 2:n
        lᵢ = A[i, 1] / δ
        A[i, 1] = lᵢ
    end
    for j in 2:n
        lⱼδ = A[j, 1] * δ
        for i in j:n
            A[i, j] -= A[i, 1] * lⱼδ'
        end
    end
    if n > 2
        _ldlt_lower!(view(A, 2:n, 2:n))
    end
    return A
end

# When the upper triangle contains the input the factorization is actially UcDU
# Only the upper triangle of the input matrix A is referenced
#
# [α aᵀ] = [1 0  ] [δ 0 ] [1 uᵀ] = [δ         δ*uᵀ     ]
# [  A₋]   [u̅ U₋'] [0 D₋] [0 U₋]   [u̅*δ u̅*δ*uᵀ + U₋'*D₋*U₋]
#
# so
#
# δ = α
# uᵀ = δ \ aᵀ
#
# followed by the recursion on the updated lower block
#
# U₋'*D₋*U₋ = A₋ - u̅*δ*uᵀ
function _ldlt_upper!(A::StridedMatrix)
    n = size(A, 2)
    δ = A[1, 1]
    for j in 2:n
        δlⱼᶜ = A[1, j]
        A[1, j] = δlⱼᶜ / δ
        for i in 2:j
            A[i, j] -= A[1, i]' * δlⱼᶜ
        end
    end
    if n > 2
        _ldlt_upper!(view(A, 2:n, 2:n))
    end
    return A
end

# When the lower triangle contains the input it is actually LDLt
# Only the lower triangle of the input matrix A is referenced
#
# [A₁₁    ] = [L₁₁  0 ] [D₁ 0 ] [L₁₁' L₂₁'] = [L₁₁*D₁*L₁₁'        L₁₁*D₁*L₂₁'       ]
# [A₂₁ A₂₂]   [L₂₁ L₂₂] [0  D₂] [0    L₂₂']   [L₂₁*D₁*L₁₁' L₂₁*D₁*L₂₁' + L₂₂*D₂*L₂₂']
#
# so
#
# L₁₁*D₁*L₁₁' = A₁₁
# L₂₁ = (A₂₁/L₁₁')/D₁
#
# and the recursion on the updated lower block
#
# L₂₂*D₂*L₂₂' = A₂₂ - L₂₁*D₁*L₂₁'
function _ldlt_lower_blocked!(A::StridedMatrix{T}, blocksize::Int, workspace::Vector{T} = Array{T}(undef, blocksize)) where T
    n = size(A, 2)
    if blocksize < n
        A₁₁ = view(A, 1:blocksize, 1:blocksize)
        _ldlt_lower!(A₁₁)
        d₁ = view(A₁₁, diagind(A₁₁))
        rdiv!(view(A, (blocksize + 1):n, 1:blocksize), UnitLowerTriangular(A₁₁)')
        rdiv!(view(A, (blocksize + 1):n, 1:blocksize), Diagonal(d₁))
        @inbounds for j in (blocksize + 1):n
            workspace .= d₁ .* conj.(view(A, j, 1:blocksize))
            for i in j:n
                for k in 1:blocksize
                    A[i, j] -= A[i, k] * workspace[k]
                end
            end
        end
        _ldlt_lower_blocked!(view(A, (blocksize + 1):n, (blocksize + 1):n), blocksize, workspace)
        return A
    else
        # If the input matrix is smaller than the chosen block size then
        # we just use the unblocked versions
        return _ldlt_lower!(A)
    end
end

# When the upper triangle contains the input the factorization is actially UcDU
# Only the upper triangle of the input matrix A is referenced
#
# [A₁₁ A₁₂] = [U₁₁' U₁₂'] [D₁ 0 ] [U₁₁ U₁₂] = [U₁₁'*D₁*U₁₁        U₁₁'*D₁*U₁₂       ]
# [    A₂₂]   [0.   U₂₂'] [0  D₂] [0   U₂₂]   [U₁₂'*D₁*U₁₁ U₁₂'*D₁*U₁₂ + U₂₂'*D₂*U₂₂]
#
# so
#
# U₁₁'*D₁*U₁₁ = A₁₁
# U₁₂ = D₁ \ (U₁₁' \ A₁₂)
#
# and the recursion on the updated lower block
#
# U₂₂' * D₂ * U₂₂ = A₂₂ - U₁₂' * D₁ * U₁₂
function _ldlt_upper_blocked!(A::StridedMatrix{T}, blocksize::Int, workspace::Vector{T} = Array{T}(undef, blocksize)) where T
    n = size(A, 2)
    if blocksize < n
        A₁₁ = view(A, 1:blocksize, 1:blocksize)
        _ldlt_upper!(A₁₁)
        d₁ = view(A₁₁, diagind(A₁₁))
        U₁₂ = view(A, 1:blocksize, (blocksize + 1):n)
        ldiv!(UnitUpperTriangular(A₁₁)', U₁₂)
        ldiv!(Diagonal(d₁), U₁₂)
        @inbounds for j in (blocksize + 1):n
            workspace .= d₁ .* view(A, 1:blocksize, j)
            for i in (blocksize + 1):j
                for k in 1:blocksize
                    A[i, j] -= A[k, i]' * workspace[k]
                end
            end
        end
        _ldlt_upper_blocked!(view(A, (blocksize + 1):n, (blocksize + 1):n), blocksize, workspace)
        return A
    else
        # If the input matrix is smaller than the chosen block size then
        # we just use the unblocked versions
        return _ldlt_upper!(A)
    end
end

"""
    ldlt!(A::Hermitian)::LTLt

See [`ldlt`](@ref)
"""
function LinearAlgebra.ldlt!(A::Hermitian{T}, blocksize::Int = max(1, 128 ÷ sizeof(T))) where T
    if A.uplo === 'U'
        _ldlt_upper_blocked!(A.data, blocksize)
    else
        _ldlt_lower_blocked!(A.data, blocksize)
    end
    return LDLt(A)
end

"""
    ldlt(A::Hermitian, blocksize::Int)::LTLt

A Hermitian LDL factorization of `A` such that `A = L*D*L'` if `A.uplo == 'L'`
and `A = U'*D*U` if `A.uplo == 'U'. Hence, the `t` is a bit of a misnomer,
but the name was introduced for real symmetric matrices where there is
no difference between the two.

Only the elements specified by `uplo` in the `Hermitian` input will be
referenced.

The factorization has three properties: `d`, `D`, and `L` which is respectively
a vector of the diagonal elements of `D`, the `Diagonal` matrix `D` and the `L`
matrix when `A.uplo == 'L'` or the `adjoint` of the `U` matrix when `A.uplo == 'U'`.

The `blocksize` argument controls the block size in the blocked algorithm.
Currently, the blocking size is set to `128 ÷ sizeof(eltype(A))`
based on very rudimentary benchmarking on my laptop. Most users won't need
adjust this argument.

# Examples
```jldoctest
julia> ldlt(Hermitian([1//1 1; 1 -1]))
LDLt{Rational{Int64}, Hermitian{Rational{Int64}, Matrix{Rational{Int64}}}}
L factor:
2×2 UnitLowerTriangular{Rational{Int64}, Adjoint{Rational{Int64}, Matrix{Rational{Int64}}}}:
 1  ⋅
 1  1
D factor:
2×2 Diagonal{Rational{Int64}, SubArray{Rational{Int64}, 1, Base.ReshapedArray{Rational{Int64}, 1, Hermitian{Rational{Int64}, Matrix{Rational{Int64}}}, Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}}}, Tuple{StepRange{Int64, Int64}}, false}}:
 1   ⋅
 ⋅  -2

julia> ldlt(Hermitian([1//1 1; 1 1], :L))
LDLt{Rational{Int64}, Hermitian{Rational{Int64}, Matrix{Rational{Int64}}}}
L factor:
2×2 UnitLowerTriangular{Rational{Int64}, Matrix{Rational{Int64}}}:
 1  ⋅
 1  1
D factor:
2×2 Diagonal{Rational{Int64}, SubArray{Rational{Int64}, 1, Base.ReshapedArray{Rational{Int64}, 1, Hermitian{Rational{Int64}, Matrix{Rational{Int64}}}, Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}}}, Tuple{StepRange{Int64, Int64}}, false}}:
 1  ⋅
 ⋅  0
```
"""
LinearAlgebra.ldlt(A::Hermitian{T}, blocksize::Int = max(1, 128 ÷ sizeof(T))) where T = ldlt!(LinearAlgebra.copy_oftype(A, typeof(oneunit(T) / one(T))))

function Base.getproperty(F::LDLt{<:Any, <:Hermitian}, d::Symbol)
    Fdata = getfield(F, :data)
    if d === :d
        return view(Fdata, diagind(Fdata))
    elseif d === :D
        return Diagonal(F.d)
    elseif d === :L
        if Fdata.uplo === 'L'
            return UnitLowerTriangular(Fdata.data)
        else
            return UnitUpperTriangular(Fdata.data)'
        end
    else
        return getfield(F, d)
    end
end

Base.copy(F::LDLt) = LDLt(copy(F.data))

function LinearAlgebra.ldiv!(F::LDLt{<:Any, <:Hermitian}, X::StridedVecOrMat)
    L = F.L
    ldiv!(L', ldiv!(F.D, ldiv!(L, X)))
end
function LinearAlgebra.lmul!(F::LDLt{<:Any, <:Hermitian}, X::StridedVecOrMat)
    L = F.L
    lmul!(L, lmul!(F.D, lmul!(L', X)))
end

(*)(F::LDLt, X::AbstractVecOrMat) = lmul!(F, copy(X))
