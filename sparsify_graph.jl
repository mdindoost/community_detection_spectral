#!/usr/bin/env julia
"""
Spectral sparsification script using Laplacians.jl

Usage: julia --project=SparsificationProject sparsify_graph.jl edges.txt n epsilon

Arguments:
  edges.txt - Path to edge list (0-based indexing, space-separated)
  n         - Number of nodes
  epsilon   - Sparsification parameter (smaller = more edges retained)

Output:
  edges_sparsified_eps{epsilon}.txt
"""

using Laplacians
using SparseArrays
using LinearAlgebra
using DelimitedFiles

function main()
    if length(ARGS) < 3
        println("Usage: julia sparsify_graph.jl edges.txt n epsilon")
        exit(1)
    end

    edges_file = ARGS[1]
    n = parse(Int, ARGS[2])
    epsilon = parse(Float64, ARGS[3])

    println("Reading edges from: $edges_file")
    println("Number of nodes: $n")
    println("Epsilon: $epsilon")

    # Read edges (0-based)
    edges_data = readdlm(edges_file, Int)
    src = edges_data[:, 1] .+ 1  # Convert to 1-based
    dst = edges_data[:, 2] .+ 1

    num_edges_original = length(src)
    println("Original edges (directed): $num_edges_original")

    # Build symmetric sparse adjacency matrix
    # Since edges are already both directions, we just create the matrix
    weights = ones(Float64, length(src))
    A = sparse(src, dst, weights, n, n)

    # Ensure symmetry (should already be symmetric for undirected graph)
    A = (A + A') / 2
    A = max.(A, 0)  # Ensure non-negative

    # Get the actual number of non-zeros (undirected edges counted once in upper triangle)
    nnz_original = nnz(A)
    undirected_edges_original = div(nnz_original, 2)
    println("Original undirected edges: $undirected_edges_original")

    # Perform spectral sparsification
    println("\nRunning spectral sparsification with epsilon=$epsilon...")
    As = sparsify(A; ep=epsilon)

    # Get statistics of sparsified graph
    nnz_sparse = nnz(As)
    undirected_edges_sparse = div(nnz_sparse, 2)

    println("Sparsified undirected edges: $undirected_edges_sparse")
    edge_ratio = undirected_edges_sparse / undirected_edges_original
    println("Edge retention ratio: $(round(edge_ratio, digits=4))")

    # Convert back to edge list (0-based)
    I, J, V = findnz(As)

    # Filter to get unique edges (avoid duplicates from symmetry)
    output_edges = []
    for k in 1:length(I)
        if I[k] < J[k]  # Only keep upper triangle
            push!(output_edges, (I[k] - 1, J[k] - 1))  # Convert to 0-based
        end
    end

    # Also include both directions for the output
    all_edges = []
    for (i, j) in output_edges
        push!(all_edges, (i, j))
        push!(all_edges, (j, i))
    end

    # Write output
    epsilon_str = replace(string(epsilon), "." => "")
    output_file = "edges_sparsified_eps$(epsilon).txt"

    open(output_file, "w") do f
        for (i, j) in all_edges
            println(f, "$i $j")
        end
    end

    println("\nOutput written to: $output_file")
    println("Edges before: $undirected_edges_original, after: $undirected_edges_sparse")
    println("Reduction: $(round((1 - edge_ratio) * 100, digits=2))%")
end

main()
