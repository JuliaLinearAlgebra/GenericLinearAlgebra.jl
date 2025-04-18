module GenericLinearAlgebra

using LinearAlgebra: LinearAlgebra,
    Adjoint, Bidiagonal, Diagonal, Factorization, Givens, HermOrSym, Hermitian, I, LowerTriangular,
    Rotation, SVD, SymTridiagonal, Symmetric, UnitLowerTriangular, UnitUpperTriangular,
    UpperTriangular,
    BLAS,
    abs2, axpy!, diag, dot, eigencopy_oftype, givens, ishermitian, mul!, rdiv!, tril, triu
using LinearAlgebra.BLAS: BlasFloat, BlasReal

include("juliaBLAS.jl")
include("lapack.jl")
include("cholesky.jl")
include("householder.jl")
include("qr.jl")
include("eigenSelfAdjoint.jl")
include("eigenGeneral.jl")
include("svd.jl")

end
