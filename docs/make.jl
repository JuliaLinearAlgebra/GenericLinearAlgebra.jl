using Documenter
using GenericLinearAlgebra

makedocs(
    sitename = "GenericLinearAlgebra",
    format = Documenter.HTML(),
    modules = [GenericLinearAlgebra]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/JuliaLinearAlgebra/GenericLinearAlgebra.jl.git"
)
