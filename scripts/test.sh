#!/bin/bash
# Quick test script for repo-classifier API
# 
# To test different repositories, edit the TEST_REPOS array below

API_URL="http://localhost:8000"

# Edit this array to test different repositories
# Format: "https://github.com/owner/repo"
TEST_REPOS=(
  "https://github.com/pandas-dev/pandas"                    # Data Science
  "https://github.com/vercel/next.js"                       # Web Development
  "https://github.com/kubernetes/kubernetes"                # DevOps
  "https://github.com/android/nowinandroid"                 # Mobile App Development
  "https://github.com/pytorch/pytorch"                      # Data Science
)

echo "🚀 Testing Repository Classifier API"
echo "====================================="
echo ""

# Test 1: Health check
echo "1️⃣  Testing health check..."
curl -s "$API_URL/" | jq '.'
echo ""
echo ""

# Test 2: Free tier (first 5 requests)
echo "2️⃣  Testing FREE TIER (first 5 requests with different repositories)..."
for i in {1..5}; do
  # Use modulo to cycle through repos if we have fewer than 5
  repo_index=$(( (i - 1) % ${#TEST_REPOS[@]} ))
  repo_url="${TEST_REPOS[$repo_index]}"
  
  echo "   Request $i: $repo_url"
  response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/classify" \
    -H "Content-Type: application/json" \
    -d "{\"repo_url\": \"$repo_url\"}")
  
  http_code=$(echo "$response" | tail -1)
  body=$(echo "$response" | sed '$d')
  
  if [ "$http_code" = "200" ]; then
    echo "$body" | jq '{status, category, confidence, rate_limit_remaining}'
  else
    echo "   ❌ HTTP $http_code"
    echo "$body" | jq '.'
  fi
  echo ""
  
  sleep 1  # Small delay between requests
done

echo ""
echo "3️⃣  Testing 6th request (should require PAYMENT - 402 response)..."
# Use the first repo for the payment test
response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/classify" \
  -H "Content-Type: application/json" \
  -d "{\"repo_url\": \"${TEST_REPOS[0]}\"}")

http_code=$(echo "$response" | tail -1)
body=$(echo "$response" | sed '$d')

if [ "$http_code" = "402" ]; then
  echo "   ✅ Correctly returned HTTP 402 (Payment Required)"
  echo "$body" | jq '.'
else
  echo "   ⚠️  Expected HTTP 402, got HTTP $http_code"
  echo "$body" | jq '.'
fi

echo ""
echo "4️⃣  Checking rate limit headers..."
curl -s -v -X POST "$API_URL/classify" \
  -H "Content-Type: application/json" \
  -d "{\"repo_url\": \"${TEST_REPOS[0]}\"}" 2>&1 | \
  grep -i "x-ratelimit" | head -5

echo ""
echo "✅ Testing complete!"

