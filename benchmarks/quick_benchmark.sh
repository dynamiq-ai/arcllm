#!/bin/bash
#
# Quick benchmark script for arcllm vs litellm
#
# Usage:
#   ./benchmarks/quick_benchmark.sh
#
# Requirements:
#   - OPENAI_API_KEY or GROQ_API_KEY set in environment or .env file
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Load .env if exists
if [ -f .env ]; then
    echo "Loading .env file..."
    set -a
    source .env
    set +a
fi

# Check for API keys
if [ -z "$OPENAI_API_KEY" ] && [ -z "$GROQ_API_KEY" ]; then
    echo "Error: No API key found. Set OPENAI_API_KEY or GROQ_API_KEY"
    exit 1
fi

# Activate venv if exists
if [ -d .venv ]; then
    source .venv/bin/activate
fi

echo "=============================================="
echo "arcllm vs litellm Quick Benchmark"
echo "=============================================="
echo ""

# Determine which provider to use
if [ -n "$GROQ_API_KEY" ]; then
    echo "Using Groq (fast API for quick benchmarks)"
    PROVIDER="groq"
    MODEL="llama-3.1-8b-instant"
else
    echo "Using OpenAI"
    PROVIDER="openai"
    MODEL="gpt-4o-mini"
fi

echo "Provider: $PROVIDER"
echo "Model: $MODEL"
echo ""

# Run benchmark
python benchmarks/benchmark_vs_litellm.py \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --iterations 5 \
    --isolated \
    --output benchmark_quick_results.json

echo ""
echo "Results saved to benchmark_quick_results.json"
