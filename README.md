# GenericLinearAlgebra.jl
[![Build Status](https://travis-ci.org/andreasnoack/GenericLinearAlgebra.jl.svg?branch=master)](https://travis-ci.org/andreasnoack/GenericLinearAlgebra.jl)
[![Build Status](https://dev.azure.com/andreasnoack/GenericLinearAlgebra/_apis/build/status/andreasnoack.GenericLinearAlgebra.jl?branchName=master)](https://dev.azure.com/andreasnoack/GenericLinearAlgebra/_build/latest?definitionId=2)
[![Coverage Status](https://coveralls.io/repos/github/andreasnoack/GenericLinearAlgebra.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/andreasnoack/GenericLinearAlgebra.jl?branch=master)

### A fresh approach to numerical linear algebra in Julia

The purpose of this package is partly to extend linear algebra functionality in base to cover generic element types, e.g. `BigFloat` and `Quaternion`, and partly to be a place to experiment with fast linear algebra routines written in Julia (except for optimized BLAS). It is my hope that it is possible to have implementations that are generic, fast, and readable.

So far, this has mainly been my playground but you might find some of the functionality here useful. The package has a generic implementation of a singular value solver (vectors not handled yet) which will make it possible to compute `norm` and `cond` of matrices of `BigFloat`. The package extends the necessary method (`svdvals!`) in base. Hence

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
