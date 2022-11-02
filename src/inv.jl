export large_inv

function block_inv(A, B, C, D_inv)
    size(A, 1) != size(A, 2) && throw(DimensionMismatch("block A is not square."))
    
    B_D_inv = B * D_inv
    pre_inv = inv(A - B_D_inv * C);
    pre_inv_B_D_inv = pre_inv * B_D_inv
    D_inv_C = D_inv * C
    B3 = -D_inv_C * pre_inv
    B4 = D_inv + D_inv_C * pre_inv_B_D_inv
        
    mat_inv = [
        pre_inv    -pre_inv_B_D_inv
        B3         B4
    ]
    return mat_inv
end


@views function partition_large_mat(mat; max_block_size)
    size(mat, 1) != size(mat, 2) && throw(DimensionMismatch("Matrix is not square."))
    Bl_1 = mat[1:max_block_size, 1:max_block_size];
    Bl_2 = mat[1:max_block_size, max_block_size+1:end];
    Bl_3 = mat[max_block_size+1:end, 1:max_block_size];
    Bl_4 = mat[max_block_size+1:end, max_block_size+1:end];
    return Bl_1, Bl_2, Bl_3, Bl_4
end


function blocks_large_mat(mat::T; max_block) where T<:AbstractMatrix{F} where F
    if size(mat, 1) <= max_block
        throw(ArgumentError("The matrix size is smaller than the specified maximum block size."))
    end
    # SubMat is the type that `partition_large_mat` returns
    SubMat = SubArray{F,2,T,Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    mat_blocks = Tuple{SubMat,SubMat,SubMat,SubMat}[]
    D = mat
    while true
        A, B, C, D = partition_large_mat(D, max_block_size = max_block)
        push!(mat_blocks, (A, B, C, D))
        size(D, 1) <= max_block && break
    end
    return mat_blocks
end


"""
    large_inv(mat; max_block_size)

Inverts a large matrix via recursive application of the matrix inversion lemma.
The matrix is recursively partitioned into blocks until those blocks are below `max_block_size`.
For large matrices it is significantly faster than `inv`.
However, it accumulates floating point error faster.
Particularly when `max_block_size` is small relative to the size of `mat`.
"""
function large_inv(mat; max_block_size)
    blocks = blocks_large_mat(mat, max_block = max_block_size);
    (A,B,C,D) = pop!(blocks)
    inverted_mat = block_inv(A, B, C, inv(D));

    @debug "iteration $iter is compeleted."
    while(!isempty(blocks))
        (A,B,C,D) = pop!(blocks)
        inverted_mat = block_inv(A, B, C, inverted_mat);
    end
    return inverted_mat
end
