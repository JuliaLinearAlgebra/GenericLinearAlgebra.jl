var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "GenericLinearAlgebra.jl",
    "title": "GenericLinearAlgebra.jl",
    "category": "page",
    "text": ""
},

{
    "location": "#LinearAlgebra.svd!",
    "page": "GenericLinearAlgebra.jl",
    "title": "LinearAlgebra.svd!",
    "category": "function",
    "text": "svd!(A[, tol, full, debug])::SVD\n\nA generic singular value decomposition (SVD). The implementation only uses Julia functions so the SVD can be computed for any element type provided that the necessary arithmetic operations are supported by the element type.\n\ntol: The relative tolerance for determining convergence. The default value is eltype(T) where T is the element type of the input matrix bidiagonal (i.e. after converting the matrix to bidiagonal form).\nfull: Sepcifies if all the left and right singular vectors be returned or if only the vectors us to the number of rows and columns of the input matrix A should be returned (the default).\ndebug: A Boolean flag to activate debug information during the executions of the algorithm. The default is false.\n\nAlgorithm\n\n...tomorrow\n\nExample\n\n\n\n\n\n\n\n"
},

{
    "location": "#LinearAlgebra.svdvals!",
    "page": "GenericLinearAlgebra.jl",
    "title": "LinearAlgebra.svdvals!",
    "category": "function",
    "text": "svdvals!(A [, tol, debug])\n\nGeneric computation of singular values.\n\njulia> using LinearAlgebra, GenericLinearAlgebra, Quaternions\n\njulia> n = 20;\n\njulia> H = [big(1)/(i + j - 1) for i in 1:n, j in 1:n]; # The Hilbert matrix\n\njulia> Float64(svdvals(H)[end]/svdvals(Float64.(H))[end] - 1) # The relative error of the LAPACK based solution in 64 bit floating point.\n-0.9999999999447275\n\njulia> A = qr([Quaternion(randn(4)...) for i in 1:3, j in 1:3]).Q *\n           Diagonal([3, 2, 1]) *\n           qr([Quaternion(randn(4)...) for i in 1:3, j in 1:3]).Q\'; # A quaternion matrix with the singular value 1, 2, and 3.\n\njulia> svdvals(A) ≈ [3, 2, 1]\ntrue\n\n\n\n\n\n"
},

{
    "location": "#GenericLinearAlgebra.jl-1",
    "page": "GenericLinearAlgebra.jl",
    "title": "GenericLinearAlgebra.jl",
    "category": "section",
    "text": "Documentation for GenericLinearAlgebra.jlCurrentModule = GenericLinearAlgebrasvd!\nsvdvals!"
},

]}
