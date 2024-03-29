# GenericLinearAlgebra.jl
[![CI](https://github.com/JuliaLinearAlgebra/GenericLinearAlgebra.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaLinearAlgebra/GenericLinearAlgebra.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JuliaLinearAlgebra/GenericLinearAlgebra.jl/branch/master/graph/badge.svg?token=eO37qmAboR)](https://codecov.io/gh/JuliaLinearAlgebra/GenericLinearAlgebra.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaLinearAlgebra.github.io/GenericLinearAlgebra.jl/stable)

### A fresh approach to numerical linear algebra in Julia

The purpose of this package is partly to extend linear algebra functionality in base to cover generic element types, e.g. `BigFloat` and `Quaternion`, and partly to be a place to experiment with fast linear algebra routines written in Julia (except for optimized BLAS). It is my hope that it is possible to have implementations that are generic, fast, and readable.

So far, this has mainly been my playground but you might find some of the functionality here useful. The package has a generic implementation of a singular value solver which will make it possible to compute `norm` and `cond` of matrices of `BigFloat`. Hence

```jl
julia> using GenericLinearAlgebra

julia> A = big.(randn(10,10));

julia> cond(A)
1.266829904721752610946505846921202851190952179974780602509001252204638657237828e+03

julia> norm(A)
6.370285271475041598951769618847832429030388948627697440637424244721679386430589
```

The package also includes functions for the blocked Cholesky and QR factorization, the self-adjoint (symmetric) and the general eigenvalue problem. These routines can be accessed by fully qualifying the names

```jl
julia> using GenericLinearAlgebra

julia> A = randn(1000,1000); A = A'A;

julia> cholesky(A);

julia> @time cholesky(A);
  0.013036 seconds (16 allocations: 7.630 MB)

julia> GenericLinearAlgebra.cholRecursive!(copy(A), Val{:L});

julia> @time GenericLinearAlgebra.cholRecursive!(copy(A), Val{:L});
  0.012098 seconds (7.00 k allocations: 7.934 MB)
```
