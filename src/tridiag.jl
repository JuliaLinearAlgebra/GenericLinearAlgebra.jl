################################################
## Specialized routines for tridiagonal matrices 
################################################

export numnegevals

"""
Computes the number of negative eigenvalues of T - σI, a.k.a. spectrum slicing

Inputs:
    T: A SymTridiagonal{<:Real} matrix
    σ: The shift parameter

Outputs:
    ν: The number of negative eigenvalues

Reference:
    B. N. Parlett, "The symmetric eigenvalue problem", Section 3.3.1, p. 52.
"""
function numnegevals{S}(T::SymTridiagonal{S}, σ::S = zero(S))
    α = T.dv
    β = T.ev
    ϵ = eps(S)
    δ = α[1] - σ
    ν = δ < 0 ? 1 : 0
    for k=1:n-1
        if δ == 0
	    info("zero in iteration $k")
	    δ = ϵ * (β[k]+ϵ) #Parlett prefers adjusting σ and starting again
	end
        δ = (α[k+1] - σ) - β[k]*(β[k]/δ)
	ν += (δ < 0)
    end
    ν
end
