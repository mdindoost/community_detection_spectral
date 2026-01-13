#!/usr/bin/env julia
"""
Spectral sparsification script using Laplacians.jl

Usage: julia --project=JuliaProject sparsify_graph.jl edges.txt n epsilon [output.txt]
"""

using Laplacians
using SparseArrays
using DelimitedFiles

function main()
    if length(ARGS) < 3
        println("Usage: julia sparsify_graph.jl edges.txt n epsilon [output.txt]")
        exit(1)
    end

    edges_file = ARGS[1]
    n = parse(Int, ARGS[2])
    epsilon = parse(Float64, ARGS[3])
    output_file = length(ARGS) >= 4 ? ARGS[4] : "edges_sparsified_eps$(epsilon).txt"

    println("Loading graph: $n nodes, epsilon=$epsilon")

    # Read edges (0-based indexing in file)
    edges_data = readdlm(edges_file, ' ', Int, '\n')
    src = edges_data[:, 1] .+ 1  # Convert to 1-based
    dst = edges_data[:, 2] .+ 1

    # Build sparse adjacency matrix
    A = sparse(src, dst, ones(Float64, length(src)), n, n)
    
    # Make symmetric
    A = A + A'
    A.nzval .= 1.0
    dropzeros!(A)
    
    undirected_edges_original = div(nnz(A), 2)
    println("Original edges: $undirected_edges_original")

    # Perform spectral sparsification
    println("Running sparsify()...")
    As = sparsify(A; ep=epsilon)
    
    undirected_edges_sparse = div(nnz(As), 2)
    println("Sparsified edges: $undirected_edges_sparse ($(round(100*undirected_edges_sparse/undirected_edges_original, digits=1))%)")

    # Extract edges
    I, J, _ = findnz(As)
    
    # Write output (both directions for each edge)
    open(output_file, "w") do f
        for k in eachindex(I)
            if I[k] < J[k]
                println(f, I[k]-1, " ", J[k]-1)
                println(f, J[k]-1, " ", I[k]-1)
            end
        end
    end

    println("Output: $output_file")
end

main()
