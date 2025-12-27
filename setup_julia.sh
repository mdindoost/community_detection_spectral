#!/bin/bash
# Setup script for Julia dependencies
# Run this once after cloning the repository

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIA_VERSION="1.10.2"

echo "=== Community Detection Spectral - Julia Setup ==="

# Check if Julia is installed
if command -v julia &> /dev/null; then
    JULIA_CMD="julia"
    echo "Found Julia: $(julia --version)"
elif [ -f "$SCRIPT_DIR/julia-$JULIA_VERSION/bin/julia" ]; then
    JULIA_CMD="$SCRIPT_DIR/julia-$JULIA_VERSION/bin/julia"
    echo "Found local Julia: $JULIA_CMD"
else
    echo "Julia not found. Installing Julia $JULIA_VERSION..."

    # Download Julia
    JULIA_URL="https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz"
    wget -q --show-progress "$JULIA_URL" -O julia.tar.gz

    # Extract
    tar -xzf julia.tar.gz
    rm julia.tar.gz

    JULIA_CMD="$SCRIPT_DIR/julia-$JULIA_VERSION/bin/julia"
    echo "Julia installed at: $JULIA_CMD"
fi

# Set up Julia depot (packages will be installed here)
export JULIA_DEPOT_PATH="$SCRIPT_DIR/julia_depot"
mkdir -p "$JULIA_DEPOT_PATH"

echo ""
echo "Installing Julia packages..."
echo "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"

# Instantiate project (install dependencies from Project.toml)
$JULIA_CMD --project="$SCRIPT_DIR/JuliaProject" -e '
    using Pkg
    println("Instantiating project...")
    Pkg.instantiate()
    println("Precompiling packages...")
    Pkg.precompile()
    println("Done!")
'

echo ""
echo "=== Julia setup complete ==="
echo ""
echo "To run sparsification manually:"
echo "  export JULIA_DEPOT_PATH=\"$JULIA_DEPOT_PATH\""
echo "  $JULIA_CMD --project=$SCRIPT_DIR/JuliaProject sparsify_graph.jl edges.txt n epsilon"
