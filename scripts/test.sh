#!/usr/bin/env bash
# Repo Classifier API Test Script
# Tests the /analyze endpoint with various repositories

set -e

BASE_URL="${BASE_URL:-http://localhost:8000}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Repo Classifier RAG Agent - API Test Suite${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Test repositories (url|expected_category)
TEST_CASES=(
  "https://github.com/solana-labs/solana|Solana"
  "https://github.com/tokio-rs/tokio|Rust"
  "https://github.com/foundry-rs/foundry|Ethereum"
  "https://github.com/denoland/deno|Deno"
  "https://github.com/pancakeswap/pancake-frontend|BNB Chain"
  "https://github.com/maticnetwork/matic.js|Polygon"
)

# Categories to send with each request
CATEGORIES='{
  "BNB Chain": "Binance Smart Chain and BNB ecosystem projects",
  "Deno": "Deno runtime, TypeScript/JavaScript server-side projects",
  "Ethereum": "Ethereum blockchain, EVM, smart contracts, and DeFi",
  "Hardhat": "Hardhat development framework and tooling projects",
  "Polygon": "Polygon network and Layer 2 scaling solutions",
  "Rust": "Rust programming language libraries and tools",
  "Solana": "Solana blockchain, programs, and ecosystem tools"
}'

# Health check
echo -e "${YELLOW}1. Health Check${NC}"
echo "   GET ${BASE_URL}/"
HEALTH=$(curl -s "${BASE_URL}/")
echo "   Response: ${HEALTH}" | head -c 200
echo ""
echo ""

# Check RAG store status
RAG_COUNT=$(echo "$HEALTH" | grep -o '"repositories":[0-9]*' | grep -o '[0-9]*' || echo "0")
if [ "$RAG_COUNT" = "0" ]; then
  echo -e "${RED}   ⚠ RAG store is empty! Run: uv run repo-classifier-ingest --input data/taxonomy.json${NC}"
  echo ""
fi

# Test each repository
echo -e "${YELLOW}2. Testing Repository Analysis${NC}"
echo ""

PASSED=0
FAILED=0

for test_case in "${TEST_CASES[@]}"; do
  # Split on pipe
  url="${test_case%%|*}"
  expected="${test_case##*|}"
  repo_name=$(echo "$url" | sed 's|https://github.com/||')
  
  echo -e "   ${BLUE}Testing: ${repo_name}${NC}"
  echo "   Expected category: ${expected}"
  
  START=$(date +%s)
  
  RESPONSE=$(curl -s -X POST "${BASE_URL}/analyze" \
    -H "Content-Type: application/json" \
    -d "{\"url\": \"${url}\", \"categories\": ${CATEGORIES}}" 2>&1)
  
  END=$(date +%s)
  DURATION=$((END - START))
  
  # Extract category from response
  CATEGORY=$(echo "$RESPONSE" | grep -o '"category":"[^"]*"' | head -1 | sed 's/"category":"//;s/"//')
  CONFIDENCE=$(echo "$RESPONSE" | grep -o '"confidence":[0-9.]*' | head -1 | sed 's/"confidence"://')
  
  if [ -n "$CATEGORY" ]; then
    if [ "$CATEGORY" = "$expected" ]; then
      echo -e "   ${GREEN}✓ PASS${NC} - Category: ${CATEGORY} (confidence: ${CONFIDENCE})"
      PASSED=$((PASSED + 1))
    else
      echo -e "   ${YELLOW}△ MISMATCH${NC} - Got: ${CATEGORY}, Expected: ${expected}"
      FAILED=$((FAILED + 1))
    fi
  else
    echo -e "   ${RED}✗ FAIL${NC} - No category returned"
    echo "   Response: ${RESPONSE}" | head -c 200
    FAILED=$((FAILED + 1))
  fi
  
  echo "   Time: ${DURATION}s"
  echo ""
done

# Summary
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Summary${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "   Passed: ${GREEN}${PASSED}${NC}"
echo -e "   Failed: ${RED}${FAILED}${NC}"
echo -e "   Total:  $((PASSED + FAILED))"
echo ""

if [ "$FAILED" -gt 0 ]; then
  exit 1
fi
