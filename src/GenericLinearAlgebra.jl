module GenericLinearAlgebra

import LinearAlgebra: mul!, ldiv!

include("juliaBLAS.jl")
include("lapack.jl")
include("cholesky.jl")
include("householder.jl")
include("qr.jl")
include("eigenSelfAdjoint.jl")
include("eigen.jl")
include("qz.jl")
include("tridiag.jl")
include("svd.jl")
include("rectfullpacked.jl")
end
