# LinearAlgebra.jl
========

### A fresh approach to numerical linear algebra in Julia

The purpose of this package is partly to extend linear algebra functionality in base to cover generic element types, e.g. `BigFloat` and `Quaternion`, and partly to be a place to experiment with fast linear algebra routines written in Julia (except for optimized BLAS). It is my hope that it is possible to have implementations that are generic, fast, and readable.

So far, this has mainly been my playground but you might find some of the functionality here useful. The package has a generic implementation of a singular value solver (vectors not handled yet) which will make it possible to compute `norm` and `cond` of matrices of `BigFloat`. The package extends the necessary method (`svdvals!`) in base. Hence

```jl
julia> using LinearAlgebra

julia> A = big(randn(10,10));

julia> cond(A)
1.266829904721752610946505846921202851190952179974780602509001252204638657237828e+03

julia> norm(A)
6.370285271475041598951769618847832429030388948627697440637424244721679386430589
```

The package also includes functions for the blocked Cholesky and QR factorization, the self-adjoint (symmetric) and the general eigenvalue problem. None of these functions are exported or extend Base methods, so, for now, the functions must be fully qualified

```jl
julia> using LinearAlgebra
A
julia> A = randn(1000,1000); A = A'A;

julia> @time cholfact(A)

julia> @time cholfact(A);
  0.013036 seconds (16 allocations: 7.630 MB)

julia> LinearAlgebra.CholeskyModule.cholRec!(copy(A), Val{:L});

julia> @time LinearAlgebra.CholeskyModule.cholRec!(copy(A), Val{:L});
  0.012098 seconds (7.00 k allocations: 7.934 MB)
```

<!-- [![StatsBase](http://pkg.julialang.org/badges/StatsBase_0.4.svg)](http://pkg.julialang.org/?pkg=StatsBase&ver=0.4) -->
[![Build Status](https://travis-ci.org/andreasnoack/LinearAlgebra.jl.svg?branch=master)](https://travis-ci.org/andreasnoack/LinearAlgebra.jl)
