module LinearAlgebra

export Sym

# Dummy type to dispatch on Symbols. Haven't found a good name for this yet.
immutable Sym{T}
end

include("juliaBLAS.jl")
include("cholesky.jl")
include("householder.jl")
include("qr.jl")
include("eigenHermitian.jl")
include("eigenGeneral.jl")

end