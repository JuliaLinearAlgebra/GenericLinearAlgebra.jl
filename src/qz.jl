
###############################################################################
# The QZ factorization is used to solve generalized eigenvalue decompositions
# of the form A x = λ B x.
# Two orthogonal matrices Q and Z are constructed to reduce the "pencil"
# `(A,B)` to another form `Q'*(A,B)*Z`:
# - in a first step `A` and `B` are reduced to upper Hessenberg and upper
#   triangular form, respectively (see: `hesstriangular`)
# - next a generalized Schur decomposition further reduces both matrices
#   to upper triangular form
# - the generalized eigenvalues are then given by the ratios of the diagonals
#   of those two triangular matrices.
###############################################################################

using LinearAlgebra:
    Givens,
    BlasInt,
    givensAlgorithm,
    eigtype,
    eigencopy_oftype,
    copy_similar

# intercept the generalized eigenvalue decomposition for generic T
LinearAlgebra.eigvals!(A::AbstractMatrix{T}, B::AbstractMatrix{T}; kwargs...) where {T <: Number} =
    generalized_eigvals!(A, B; kwargs...)

"Compute the eigenvalues of the generalized eigenvalue decomposition."
function generalized_eigvals(A::AbstractMatrix{TA}, B::AbstractMatrix{TB}; kwargs...) where {TA,TB}
    S = promote_type(eigtype(TA),TB)
    generalized_eigvals!(eigencopy_oftype(A, S), eigencopy_oftype(B, S); kwargs...)
end

# in-place implementation
function generalized_eigvals!(A::AbstractArray{<:Real}, B::AbstractArray{<:Real};
    permute::Bool = false, scale::Bool = false,
    sortby::Union{Function,Nothing} = LinearAlgebra.eigsortby)
    @assert size(A,1) == size(A,2) == size(B,1) == size(B,2)

    n = size(A,1)
    T = eltype(A)

    # reduce pencil to Hessenberg-triangular form, don't store Q and Z
    hesstriangular!(A, B, I, I; computeQ=false, computeZ=false)
    # use double shift QZ here to avoid complex arithmetic
    qz_double!(A, B, I, I;
        computeQ=false, computeZ=false, computeST=false)

    # now we have to check for complex eigenvalues
    # if there are any, the subdiagonal of A is non-zero and the eigenvalues
    # are the eigenvalues of a 2x2 block
    complexeig = false
    for k in 1:n-1
        if A[k+1,k] != 0
            complexeig = true
            break
        end
    end
    if !complexeig
        # eigenvalues are ratios of diagonal elements
        eigvals = diag(A) ./ diag(B)
        LinearAlgebra.sorteig!(eigvals, sortby)
    else
        # there may be 2x2 subblocks
        eigvals = zeros(Complex{T}, n)
        k = 1
        while k <= n
            if (k < n) && (A[k+1,k] != 0)
                αr, αi, δ1, βr, βi, δ2 = eig22(view(A,k:k+1,k:k+1), view(B,k:k+1,k:k+1))
                eigvals[k] = (αr + im*αi) / δ1
                eigvals[k+1] = (βr + im*βi) / δ2
                k += 2
            else
                eigvals[k] = A[k,k] / B[k,k]
                k += 1
            end
        end
        LinearAlgebra.sorteig!(eigvals, sortby)
    end
end

function generalized_eigvals!(
    A::AbstractArray{<:Complex}, B::AbstractArray{<:Complex};
    sortby::Union{Function,Nothing} = LinearAlgebra.eigsortby,
    kwargs...)

    @assert size(A,1) == size(A,2) == size(B,1) == size(B,2)

    n = size(A,1)
    T = eltype(A)

    hesstriangular!(A, B, I, I; computeQ=false, computeZ=false)
    qz_single!(A, B, I, I;
        computeQ=false, computeZ=false, computeST=false)
    LinearAlgebra.sorteig!(diag(A) ./ diag(B), sortby)
end


## Common routines

# matrix-matrix product with a workspace provided
function mat_lmul!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, workspace::AbstractVector{T}) where {T<:Number}
    m,n,lw = size(A,1), size(B,2), length(workspace)
    if lw < m*n
        throw(DimensionMismatch("provided workspace is too small"))
    end
    temp = reshape(view(workspace,1:m*n),m,n)
    mul!(temp,A,B)
    # temp = A*B
    B .= temp
end

function mat_rmul!(A::AbstractMatrix, B::AbstractMatrix, workspace::AbstractVector)
    m,n,lw = size(A,1), size(B,2), length(workspace)
    if lw < m*n
        throw(DimensionMismatch("provided workspace is too small"))
    end
    temp = reshape(view(workspace,1:m*n),m,n)
    mul!(temp,A,B)
    # temp = A*B
    A .= temp
end

# compute eigenvalues of a 2x2 matrix, used to obtain a shift in qz_single
function eig22(A::AbstractMatrix{Complex{T}}, B::AbstractMatrix{Complex{T}}, e::Complex{T}) where {T<:AbstractFloat}
    detb = B[1, 1] * B[2, 2] - B[1, 2] * B[2, 1]

    ab11 = A[1, 1] * B[2, 2] - A[2, 1] * B[1, 2]
    ab12 = A[1, 2] * B[2, 2] - A[2, 2] * B[1, 2]
    ab21 = A[2, 1] * B[1, 1] - A[1, 1] * B[2, 1]
    ab22 = A[2, 2] * B[1, 1] - A[1, 2] * B[2, 1]

    t = one(T) / 2 * (ab11 + ab22)
    d = t^2 + ab12 * ab21 - ab11 * ab22
    d = sqrt(d)

    e1 = (t + d) / detb
    e2 = (t - d) / detb

    abs2(e1 - e) < abs2(e2 - e) ? e1 : e2
end

# compute both generalized eigenvalues of 2x2 matrices A and B
# the result is αr,αi,δ1,βr,βi,δ2, and the corresponding eigenvalues are
# (αr+im*αi)/δ1 and (βr+im*βi)/δ1
function eig22(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:AbstractFloat}
    detb = B[1, 1] * B[2, 2] - B[1, 2] * B[2, 1]

    ab11 = A[1, 1] * B[2, 2] - A[2, 1] * B[1, 2]
    ab12 = A[1, 2] * B[2, 2] - A[2, 2] * B[1, 2]
    ab21 = A[2, 1] * B[1, 1] - A[1, 1] * B[2, 1]
    ab22 = A[2, 2] * B[1, 1] - A[1, 2] * B[2, 1]

    t = one(T) / 2 * (ab11 + ab22)
    d = t^2 + ab12 * ab21 - ab11 * ab22
    if d >= 0
        d = sqrt(d)
        return t+d,zero(T),detb,t-d,zero(T),detb
    else
        d = sqrt(-d)
        return t,d,detb,t,-d,detb
    end
end

## Householder reflections

"""
    floatmin_maxinv(T)

Smallest floating point number such that 1/(x*eps(T)) can be exactly
represented for numbers of type `T`.
"""
floatmin_maxinv(::Type{Float32}) = reinterpret(Float32, 0x26000000)
floatmin_maxinv(::Type{Float64}) = reinterpret(Float64, 0x21a0000000000000)
# This is just a crude guess for generic types T, specialize if necessary.
floatmin_maxinv(::Type{T}) where {T <: AbstractFloat} = 1e7*sqrt(floatmin(T))

# Householder transform for a 3x3 matrix
function householderAlgorithm3(alpha::T, x1::T, x2::T) where {T<:AbstractFloat}
    onepar = one(T)
    twopar = 2one(T)
    zeropar = zero(T)
    safmn2 = floatmin_maxinv(T)
    safmx2 = one(T)/safmn2

    xnorm = hypot(x1,x2)

    if xnorm == zeropar
        beta = alpha
        tau = zeropar
        v1 = x1
        v2 = x2
    else
        if alpha == 0
            beta = -hypot(alpha, xnorm)
        else
            beta = -sign(alpha)*hypot(alpha, xnorm)
        end
        count = 0
        while abs(beta) < safmn2 && count < 20
            x1 = x1*safmx2
            x2 = x2*safmx2
            beta = beta*safmx2
            alpha = alpha*safmx2
            count = count + 1
        end
        xnorm = hypot(x1,x2)
        tau = (beta-alpha)/beta
        v1 = x1/(alpha-beta)
        v2 = x2/(alpha-beta)
        for i in 1:count
            beta = beta*safmn2
        end
    end
    beta, v1, v2, tau
end

# Opposite householder transform for 3x3 matrix
function oppositeHouseholderAlgorithm3(A::AbstractMatrix)
    # Calculate a sequence of givens rotations forming the RQ decomposition of A
    c1,s1,r1 = givensAlgorithm(A[3,2],A[3,1])
    c2,s2,r2 = givensAlgorithm(A[3,3],r1)
    t1 = -A[2,2]*s1 + A[2,1]*c1
    t2 = -s2*A[2,3] + c2*(c1*A[2,2] + s1*A[2,1])
    c3,s3,r3 = givensAlgorithm(t2,t1)

    # Solve A\e1
    scale = c3*(A[1,1]*c1 - A[1,2]*s1) + s3*(A[1,3]*s2 - c2*(A[1,2]*c1 + A[1,1]*s1))
    T = eltype(A)
    if abs(scale) < 100eps(T)
        householderAlgorithm3(one(T),zero(T),zero(T))
    else
        scale = 1/scale
        t1 = scale*(c1*c3 - c2*s1*s3)
        t2 = -scale*(c3*s1 + c1*c2*s3)
        t3 = scale*s2*s3

        householderAlgorithm3(t1,t2,t3)
    end
end

# Opposite householder transform for 2x3 matrix
function oppositeHouseholderAlgorithm3_r(A::AbstractMatrix)
    # Calculate a sequence of givens rotations forming the RQ decomposition of A
    c1,s1,r1 = givensAlgorithm(A[2,2],A[2,1])
    c2,s2,r2 = givensAlgorithm(A[2,3],r1)
    t1 = -A[1,2]*s1 + A[1,1]*c1
    t2 = -s2*A[1,3] + c2*(c1*A[1,2] + s1*A[1,1])
    c3,s3,r3 = givensAlgorithm(t2,t1)

    t1 = c1*c3 - c2*s1*s3
    t2 = -(c3*s1 + c1*c2*s3)
    t3 = s2*s3

    householderAlgorithm3(t1,t2,t3)
end

function house3_apply_l!(A::AbstractMatrix, tau, v1, v2)
    @assert size(A,1) == 3
    n = size(A,2)
    @inbounds for i in 1:n
        temp = tau*(A[1,i] + A[2,i]*v1 + A[3,i]*v2)
        A[1,i] = A[1,i] - temp
        A[2,i] = A[2,i] - temp*v1
        A[3,i] = A[3,i] - temp*v2
    end
end

function house3_apply_r!(A::AbstractMatrix, tau, v1, v2)
    @assert size(A,2) == 3
    n = size(A,1)
    @inbounds for i in 1:n
        temp = tau*(A[i,1] + A[i,2]*v1 + A[i,3]*v2)
        A[i,1] = A[i,1] - temp
        A[i,2] = A[i,2] - temp*v1
        A[i,3] = A[i,3] - temp*v2
    end
end

## Givens rotations and helper functions

"Apply the Givens rotation to a range of columns of `A`."
function givens_lmul!(G::Givens, A::AbstractMatrix, j1::Integer, j2::Integer)
    m, n = size(A, 1), size(A, 2)
    if G.i2 > m
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end
    @inbounds for i in j1:j2
        temp = G.c * A[G.i1, i] + G.s * A[G.i2, i]
        A[G.i2, i] = -conj(G.s) * A[G.i1, i] + G.c * A[G.i2, i]
        A[G.i1, i] = temp
    end
    A
end

"Apply the Givens rotation to a range of columns of `A`."
function givens_rmul!(A::AbstractMatrix, G::Givens, j1::Integer, j2::Integer)
    m, n = size(A, 1), size(A, 2)
    if G.i2 > n
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end
    @inbounds for i in j1:j2
        temp = A[i, G.i1] * G.c - A[i, G.i2] * conj(G.s)
        A[i, G.i2] = A[i, G.i1] * G.s + A[i, G.i2] * G.c
        A[i, G.i1] = temp
    end
    A
end

# Direct implementation of givens using c and s for complex matrices
function givens_apply_l!(A::AbstractMatrix{Complex{T}},
    c::T, s::Complex{T},
    i1::Integer, i2::Integer, j1::Integer, j2::Integer) where {T<:AbstractFloat}

    m, n = size(A, 1), size(A, 2)
    if i1 > m || i2 > m || i1 < 1 || i2 < 1
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end
    @inbounds for i in j1:j2
        temp = c * A[i1, i] + s * A[i2, i]
        A[G.i2, i] = -conj(s) * A[i1, i] + c * A[i2, i]
        A[G.i1, i] = temp
    end
end

function givens_apply_r!(A::AbstractMatrix{Complex{T}},
    c::T, s::Complex{T},
    i1::Integer, i2::Integer, j1::Integer, j2::Integer) where {T<:AbstractFloat}

    @inbounds for i in j1:j2
        temp = A[i, i1] * c - A[i, i2] * conj(s)
        A[i, i2] = A[i, i1] * s + A[i, i2] * c
        A[i, i1] = temp
    end
end

# Direct implementation of givens using c and s for real matrices
function givens_apply_l!(v1::AbstractVector{T}, v2::AbstractVector{T},
        c::T, s::T) where {T<:AbstractFloat}
    n = length(v1)
    @inbounds for i in 1:n
        temp = v1[i]
        v1[i] = c * temp + s * v2[i]
        v2[i] = -s * temp + c * v2[i]
    end
end

function givens_apply_r!(v1::AbstractVector{T}, v2::AbstractVector{T},
        c::T, s::T) where {T<:AbstractFloat}
    n = length(v1)
    @inbounds @simd for i in 1:n
        temp = v1[i]
        v1[i] = temp * c - v2[i]*s
        v2[i] = temp * s + v2[i]*c
    end
end


## QZ specific helper routines

# move double shift bulge one position down (swap double pole and infinite pole)
function swappoles_double_inf(
        A::AbstractMatrix, B::AbstractMatrix,
        Q, Z,
        k::Integer, istartm::Integer, istopm::Integer;
        computeQ::Bool = true, computeZ::Bool = true,
        qStart::Integer = 1, zStart::Integer = 1
    )

    beta, v1, v2, tau = oppositeHouseholderAlgorithm3(view(B,k:k+2,k:k+2))
    house3_apply_r!(view(A,istartm:k+3,k:k+2),tau,v1,v2)
    house3_apply_r!(view(B,istartm:k+2,k:k+2),tau,v1,v2)
    # B[k,k] = beta # This formula is not correct for opposite householder
    B[k+1,k] = 0
    B[k+2,k] = 0
    if computeZ
        house3_apply_r!(view(Z,:,k-zStart+1:k-zStart+3),tau,v1,v2)
    end

    beta, v1, v2, tau = householderAlgorithm3(A[k+1,k],A[k+2,k],A[k+3,k])
    house3_apply_l!(view(A,k+1:k+3,k:istopm),tau,v1,v2)
    house3_apply_l!(view(B,k+1:k+3,k:istopm),tau,v1,v2)
    A[k+1,k] = beta
    A[k+2,k] = 0
    A[k+3,k] = 0
    if computeQ
        house3_apply_r!(view(Q,:,k-qStart+2:k-qStart+4),tau,v1,v2)
    end
end

# introduce a double shift bulge at the start of the pencil
function setdoubleshift(
        A::AbstractMatrix{T}, B::AbstractMatrix{T},
        Q,
        alphar1::T, alphai1::T, beta1::T,
        alphar2::T, alphai2::T, beta2::T,
        ilo::Integer, istopm::Integer,
        computeQ::Bool, qStart::Integer = 1
    ) where {T <: AbstractFloat}

    # Calculate v = (beta2*A - alphar2*B)*inv(B)*(beta1*A-alphar1*B)*e1 - alphai1*alphai2*B*e1
    v = beta1*A[ilo:ilo+2,ilo] - alphar1*B[ilo:ilo+2,ilo]
    v[2] = v[2]/B[ilo+1,ilo+1]
    v[1] = (v[1] - B[ilo,ilo+1]*v[2])/B[ilo,ilo]
    v = beta2*A[ilo:ilo+2,ilo]*v[1] +
        beta2*A[ilo:ilo+2,ilo+1]*v[2] -
        alphar2*(B[ilo:ilo+2,ilo]*v[1] + B[ilo:ilo+2,ilo+1]*v[2])
    v[1] = v[1] - alphai1*alphai2*B[ilo,ilo]

    # Calculate and apply householder reflector
    beta, v1, v2, tau = householderAlgorithm3(v[1],v[2],v[3])
    house3_apply_l!(view(A,ilo:ilo+2,ilo:istopm),tau,v1,v2)
    house3_apply_l!(view(B,ilo:ilo+2,ilo:istopm),tau,v1,v2)
    if computeQ
        house3_apply_r!(view(Q,:,ilo-qStart+1:ilo-qStart+3),tau,v1,v2)
    end
end

# remove a double shift bulge at the end of the pencil
function removedoubleshift(
        A::AbstractMatrix, B::AbstractMatrix,
        Q, Z,
        istartm::Integer, istopm::Integer, ihi::Integer;
        computeQ::Bool = true, qStart::Integer = 1,
        computeZ::Bool = true, zStart::Integer = 1
    )

    beta, v1, v2, tau = oppositeHouseholderAlgorithm3(view(B,ihi-2:ihi,ihi-2:ihi))
    house3_apply_r!(view(A,istartm:ihi,ihi-2:ihi),tau,v1,v2)
    house3_apply_r!(view(B,istartm:ihi,ihi-2:ihi),tau,v1,v2)
    B[ihi-1,ihi-2] = 0
    B[ihi,ihi-2] = 0
    if computeZ
        house3_apply_r!(view(Z,:,ihi-2-zStart+1:ihi-zStart+1),tau,v1,v2)
    end

    c,s,beta = givensAlgorithm(A[ihi-1,ihi-2],A[ihi,ihi-2])
    givens_apply_l!(view(A,ihi-1,ihi-1:istopm),view(A,ihi,ihi-1:istopm),c,s)
    givens_apply_l!(view(B,ihi-1,ihi-1:istopm),view(B,ihi,ihi-1:istopm),c,s)
    A[ihi-1,ihi-2] = beta
    A[ihi,ihi-2] = 0
    if computeQ
        givens_apply_r!(view(Q,:,ihi-1-qStart+1),view(Q,:,ihi-qStart+1),c,-s)
    end

    c,s,beta = givensAlgorithm(B[ihi,ihi],B[ihi,ihi-1])
    givens_apply_r!(view(A,istartm:ihi,ihi),view(A,istartm:ihi,ihi-1),c,-s)
    givens_apply_r!(view(B,istartm:ihi,ihi),view(B,istartm:ihi,ihi-1),c,-s)
    B[ihi,ihi] = beta
    B[ihi,ihi-1] = 0
    if computeZ
        givens_apply_r!(view(Z,:,ihi-zStart+1),view(Z,:,ihi-1-zStart+1),c,-s)
    end
end


## Reduction to Hessenberg-triangular form

"""
    A1,B1,Q1,Z1 = hesstriangular(A, B[, Q, Z])

Reduce the given `(A,B)` pencil to Hessenberg-triangular form.

The result is an equivalent pencil `(A1,B1)` in which `A1` is upper Hessenberg
and `B1` is upper triangular. The pencils satisfy `Q1*(A1,B1)*Z1' = (A,B)`.

If pencil matrices `Q` and `Z` are given, then the output pencil satisfies
`Q1*(A1,B1)*Z1' = Q*(A,B)*Z'`. That is, the matrices `Q` and `Z` are updated
with the transformation to Hessenberg-triangular form.
"""
function hesstriangular(A, B)
    S = promote_type(eltype(A), eltype(B))
    n = size(A,1)
    Q = Matrix(one(S)*I, n, n)
    Z = Matrix(one(S)*I, n, n)
    hesstriangular(A, B, Q, Z)
end

function hesstriangular(A, B, Q, Z)
    # make a dense copy of A and B
    S = promote_type(eltype(A), eltype(B))
    A1 = copy_similar(A, S)
    B1 = copy_similar(B, S)
    hesstriangular!(A1, B1, Q, Z)
    UpperHessenberg(A1), UpperTriangular(B1), Q, Z
end

# inplace implementation: all matrices are overwritten
function hesstriangular!(
        A::AbstractMatrix, B::AbstractMatrix,
        Q, Z;
        computeQ::Bool = true, computeZ::Bool = true
    )

    @assert size(A,1)==size(A,2)==size(B,1)==size(B,2)
    computeQ && @assert size(Q,1)==size(Q,2)
    computeQ && @assert size(A,1)==size(Q,1)
    computeZ && @assert size(Z,1)==size(Z,2)
    computeZ && @assert size(A,1)==size(Z,1)

    n = size(A,1)

    # reduce B to upper-triangular (with QR)
    for i in 1:n
        for j in n-1:-1:i
            G,r = givens(B,j,j+1,i)
            B[j,i] = r
            B[j+1,i] = 0
            givens_lmul!(G,B,i+1,n)
            lmul!(G,A)
            computeQ && rmul!(Q,adjoint(G))
        end
    end
    # reduce A to upper-hessenberg while keeping B upper triangular
    for i in 1:n
        for j in n-1:-1:i+1
            G,r = givens(A,j,j+1,i)
            A[j,i] = r
            A[j+1,i] = 0
            givens_lmul!(G,B,j,n)
            givens_lmul!(G,A,i+1,n)
            computeQ && rmul!(Q,adjoint(G))

            G,r = givens(B[j+1,j+1],B[j+1,j],j,j+1)
            B[j+1,j] = 0
            B[j+1,j+1] = r
            givens_rmul!(B,G,1,j)
            rmul!(A,G)
            computeZ && rmul!(Z,G)
        end
    end
end


## the QZ factorization

qz(A, B) = qz_single(A, B)

function qz_single(A::AbstractMatrix, B::AbstractMatrix; kwargs...)
    @assert size(A,1)==size(A,2)==size(B,1)==size(B,2)
    n = size(A,1)
    ELT = promote_type(eltype(A),eltype(B))
    Q = Matrix{ELT}(I, n, n)
    Z = similar(Q)
    S = LinearAlgebra.copy_similar(A, ELT)
    T = LinearAlgebra.copy_similar(B, ELT)
    qz_single!(S, T, Q, Z; computeQ=true, computeZ=true, computeST=true)
    UpperTriangular(S), UpperTriangular(T), Q, Z
end

qz_single!(A, B, Q, Z; kwargs...) =
    qz_single!(A, B, Q, Z, 1, size(A,1); kwargs...)

"""
Apply the single shift QZ algorithm in place. The matrices `A` and `B` are
used as workspace.

Options:
- computeQ: construct the Q matrix (not necessary if only eigenvalues are needed)
- computeZ: construct the Z matrix (not necessary if only eigenvalues are needed)
- computeST: compute the full triangular `S` and `T` matrices of the stencil
  after QZ reduction. If computeST is false, then only the diagonal entries (or
  small diagonal blocks) are computed, which is enough to compute the eigenvalues.
"""
function qz_single!(
        A::AbstractMatrix, B::AbstractMatrix,
        Q, Z,
        is::Integer, ie::Integer;
        computeST::Bool = true, computeQ::Bool = true, computeZ::Bool = true,
        qStart::Integer = 1, zStart::Integer = 1
    )

    n = size(A,1)
    T = eltype(A)

    ilo = is
    ihi = ie
    tol = eps(real(T))

    bnorm = norm(B)

    nb_swaps = 0
    nb_iterations = 0
    maxit = 30*n
    ld = 0
    eshift = zero(complex(T))
    for iter in 1:maxit

        if abs(A[ihi,ihi-1]) <= tol*(abs(A[ihi,ihi]) + abs(A[ihi-1,ihi-1]))
            A[ihi,ihi-1] = 0
            ihi = ihi - 1
            ld = 0
            eshift = zero(complex(T))
        end

        if ilo >= ihi
            break
        end

        # Check for interior deflation
        istart = ilo
        for k in ihi-1:-1:ilo+1
            if abs(A[k,k-1]) <= tol*(abs(A[k,k]) + abs(A[k-1,k-1]))
                A[k,k-1] = 0
                istart = k
                break
            end
        end
        # Check for infinite eigenvalues
        if computeST
            istartm = 1
            istopm = n
        else
            istartm = istart
            istopm = ihi
        end
        for k in istart:ihi
            if abs(B[k,k]) <= tol*bnorm
                B[k,k] = 0
                for k2 in k:-1:istart+1
                    G,r = givens(B[k2-1,k2],B[k2-1,k2-1],k2-1,k2)
                    B[k2-1,k2] = r
                    B[k2-1,k2-1] = 0
                    givens_rmul!(B,G,istartm,k2-2)
                    givens_rmul!(A,G,istartm,k2+2)
                    if computeZ
                        G = Givens(k2-zStart, k2-zStart+1, G.c, G.s)
                        rmul!(Z,G)
                    end

                    if k2 < ihi
                        G,r = givens(A,k2,k2+1,k2-1)
                        A[k2,k2-1] = r
                        A[k2+1,k2-1] = 0
                        givens_lmul!(G,A,k2,istopm)
                        givens_lmul!(G,B,k2,istopm)
                        if computeQ
                            G = Givens(k2-qStart+1, k2-qStart+2,G.c, G.s)
                            rmul!(Q,adjoint(G))
                        end
                    end
                end
                if istart < ihi
                    G,r = givens(A,istart,istart+1,istart)
                    A[istart,istart] = r
                    A[istart+1,istart] = 0
                    givens_lmul!(G,A,istart+1,istopm)
                    givens_lmul!(G,B,istart+1,istopm)
                    if computeQ
                        G = Givens(istart-qStart+1, istart-qStart+2,G.c, G.s)
                        rmul!(Q,adjoint(G))
                    end
                end

                istart = istart + 1
            end
        end

        if istart >= ihi
            continue
        end

        if computeST
            istartm = 1
            istopm = n
        else
            istartm = istart
            istopm = ihi
        end

        nb_iterations = nb_iterations + 1
        ld = ld + 1

        # Get shift
        if mod(ld,10) !== 0
            shift = eig22(view(A,ihi-1:ihi,ihi-1:ihi),view(B,ihi-1:ihi,ihi-1:ihi), A[ihi,ihi]/B[ihi,ihi])
        else
            eshift = eshift + A[ihi, ihi-1]/B[ihi-1,ihi-1]
            shift = eshift
        end

        # Introduce shift
        h1 = A[istart,istart] - shift * B[istart,istart]
        G,r = givens(h1,A[istart+1,istart],istart,istart+1)
        givens_lmul!(G,A,istart,istopm)
        givens_lmul!(G,B,istart,istopm)
        if computeQ
            G = Givens(istart-qStart+1, istart-qStart+2,G.c,G.s)
            rmul!(Q,adjoint(G))
        end

        for k in istart:ihi-2
            G,r = givens(B[k+1,k+1],B[k+1,k],k,k+1)
            B[k+1,k+1] = r
            B[k+1,k] = 0
            givens_rmul!(B,G,istartm,k)
            givens_rmul!(A,G,istartm,k+2)
            if computeZ
                G = Givens(k-zStart+1, k-zStart+2,G.c, G.s)
                rmul!(Z,G)
            end

            G,r = givens(A,k+1,k+2,k)
            A[k+1,k] = r
            A[k+2,k] = 0
            givens_lmul!(G,A,k+1,istopm)
            givens_lmul!(G,B,k+1,istopm)
            if computeQ
                G = Givens(k-qStart+2, k-qStart+3,G.c, G.s)
                rmul!(Q,adjoint(G))
            end
            nb_swaps = nb_swaps + 1
        end

        # Remove shift
        G,r = givens(B[ihi,ihi], B[ihi,ihi-1],ihi-1,ihi)
        givens_rmul!(A,G,istartm,ihi)
        givens_rmul!(B,G,istartm,ihi-1)
        B[ihi,ihi] = r
        B[ihi,ihi-1] = zero(T)
        if computeZ
            G = Givens(ihi-zStart, ihi-zStart+1,G.c, G.s)
            rmul!(Z,G)
        end

    end
    return nb_iterations, nb_swaps
end


## qz_double

function qz_double(A::AbstractMatrix, B::AbstractMatrix; kwargs...)
    @assert size(A,1)==size(A,2)==size(B,1)==size(B,2)
    n = size(A,1)
    ELT = promote_type(eltype(A),eltype(B))
    Q = Matrix{ELT}(I, n, n)
    Z = similar(Q)
    S = LinearAlgebra.copy_similar(A, ELT)
    T = LinearAlgebra.copy_similar(B, ELT)
    qz_double!(S, T, Q, Z; computeQ=true, computeZ=true, computeST=true)
    # S and T may not be triangular due to subdiagonal entries
    UpperHessenberg(S), UpperHessenberg(T), Q, Z
end

qz_double!(A, B, Q, Z; kwargs...) =
    qz_double!(A, B, Q, Z, 1, size(A,1); kwargs...)

function qz_double!(
        A::AbstractMatrix, B::AbstractMatrix,
        Q, Z,
        is::Integer, ie::Integer;
        computeST::Bool = true, computeQ::Bool = true, computeZ::Bool = true,
        qStart::Integer = 1, zStart::Integer = 1
    )

    n = size(A,1)
    T = eltype(A)

    ilo = is
    ihi = ie
    tol = eps(real(T))

    bnorm = norm(B)

    nb_swaps = 0
    nb_iterations = 0
    maxit = 30*n
    ld = 0
    eshift = zero(T)
    for iter in 1:maxit
        if ilo + 1 >= ihi
            break
        end
        if abs(A[ihi,ihi-1]) <= tol*(abs(A[ihi,ihi]) + abs(A[ihi-1,ihi-1]))
            A[ihi,ihi-1] = 0
            ihi = ihi - 1
            ld = 0
            eshift = zero(T)
        end
        if ilo + 1 >= ihi
            break
        end
        if abs(A[ihi-1,ihi-2]) <= tol*(abs(A[ihi-1,ihi-1]) + abs(A[ihi-2,ihi-2]))
            A[ihi-1,ihi-2] = 0
            ihi = ihi - 2
            ld = 0
            eshift = zero(T)
        end

        if ilo + 1 >= ihi
            break
        end

        # Check for interior deflation
        istart = ilo
        for k in ihi-2:-1:ilo+1
            if abs(A[k,k-1]) <= tol*(abs(A[k,k]) + abs(A[k-1,k-1]))
                A[k,k-1] = 0
                istart = k
                break
            end
        end
        # Check for infinite eigenvalues
        if computeST
            istartm = 1
            istopm = n
        else
            istartm = istart
            istopm = ihi
        end
        for k in istart:ihi
            if abs(B[k,k]) <= tol*bnorm
                B[k,k] = 0
                for k2=k:-1:istart+1
                    G,r = givens(B[k2-1,k2],B[k2-1,k2-1],k2-1,k2)
                    B[k2-1,k2] = r
                    B[k2-1,k2-1] = 0
                    givens_rmul!(B,G,istartm,k2-2)
                    givens_rmul!(A,G,istartm,k2+2)
                    if computeZ
                        G = Givens(k2-zStart, k2-zStart+1,G.c, G.s)
                        rmul!(Z,G)
                    end

                    if k2 < ihi
                        G,r = givens(A,k2,k2+1,k2-1)
                        A[k2,k2-1] = r
                        A[k2+1,k2-1] = 0
                        givens_lmul!(G,A,k2,istopm)
                        givens_lmul!(G,B,k2,istopm)
                        if computeQ
                            G = Givens(k2-qStart+1, k2-qStart+2,G.c, G.s)
                            rmul!(Q,adjoint(G))
                        end
                    end
                end
                if istart < ihi
                    G,r = givens(A,istart,istart+1,istart)
                    A[istart,istart] = r
                    A[istart+1,istart] = 0
                    givens_lmul!(G,A,istart+1,istopm)
                    givens_lmul!(G,B,istart+1,istopm)
                    if computeQ
                        G = Givens(istart-qStart+1, istart-qStart+2,G.c, G.s)
                        rmul!(Q,adjoint(G))
                    end
                end

                istart = istart + 1
            end
        end

        if istart + 1 >= ihi
            continue
        end

        if computeST
            istartm = 1
            istopm = n
        else
            istartm = istart
            istopm = ihi
        end

        nb_iterations = nb_iterations + 2
        ld = ld + 1

        # Get shift
        if mod(ld,10) !== 0
            shift1r,shift1i,shift1s,shift2r,shift2i,shift2s =
                eig22(view(A,ihi-1:ihi,ihi-1:ihi),view(B,ihi-1:ihi,ihi-1:ihi))
        else
            eshift = eshift + A[ihi, ihi-1]/B[ihi-1,ihi-1]
            shift1r = eshift
            shift1i = zero(T)
            shift1s = one(T)
            shift2r = -eshift
            shift2i = zero(T)
            shift2s = one(T)
        end

        # Introduce shift
        setdoubleshift(A, B, Q,
            shift1r, shift1i, shift1s,
            shift2r, shift2i, shift2s,
            istart, istopm, computeQ, qStart)

        for k in istart:ihi-3
            swappoles_double_inf(A, B, Q, Z,
                k, istartm, istopm; computeQ,computeZ,qStart,zStart)
            nb_swaps = nb_swaps + 2
        end

        # Remove double shift
        removedoubleshift(A,B,Q,Z,istartm,istopm,ihi; computeQ,qStart,computeZ,zStart)
    end

    # Standardize
    k = is
    while k < ie
        if A[k+1,k] != 0
            er1,ei1,es1,er2,ei2,es2 = eig22(view(A,k:k+1,k:k+1),view(B,k:k+1,k:k+1))
            if ei1 == 0
                # Bulge has real eigenvalues, make upper triangular
                if computeST
                    istartm = 1
                    istopm = n
                else
                    istartm = k
                    istopm = k+1
                end

                for i in 1:3
                    if abs(A[k+1,k]) <= tol*( abs(A[k,k]) + abs(A[k+1,k+1]) )
                        A[k+1,k] = 0
                        break
                    end

                    c,s,r = givensAlgorithm(es1*A[k,k] - er1*B[k,k],es1*A[k+1,k])
                    givens_apply_l!(view(A,k,k:istopm),view(A,k+1,k:istopm),c,s)
                    givens_apply_l!(view(B,k,k:istopm),view(B,k+1,k:istopm),c,s)
                    if computeQ
                        givens_apply_r!(view(Q,:,k-qStart+1),view(Q,:,k-qStart+2),c,-s)
                    end

                    c,s,r = givensAlgorithm(B[k+1,k+1],B[k+1,k])
                    givens_apply_r!(view(A,istartm:k+1,k+1),view(A,istartm:k+1,k),c,-s)
                    givens_apply_r!(view(B,istartm:k+1,k+1),view(B,istartm:k+1,k),c,-s)
                    if computeZ
                        givens_apply_r!(view(Z,:,k-zStart+2),view(Z,:,k-zStart+1),c,-s)
                    end
                    B[k+1,k] = zero(T)
                end

            end
            k = k + 2
        else
            k = k + 1
        end
    end

    return nb_iterations, nb_swaps
end
