# Repository Classifier API

A FastAPI service that automatically classifies GitHub repositories into categories using zero-shot machine learning. Features a freemium model with 5 free requests per minute, then requires payment via x402 protocol for additional requests.

## Features

- **Zero-Shot Classification**: Uses DeBERTa-v3 model for accurate repository categorization
- **README Analysis**: Fetches and analyzes repository README files for better context
- **Freemium Model**: 5 free requests per minute, then $0.01 USD per request via x402
- **Rate Limiting**: Built-in rate limiting with clear headers
- **RESTful API**: Both GET and POST endpoints with OpenAPI documentation
- **Docker Ready**: Fully containerized and production-ready

## Categories

Repositories are classified into one of these categories:
- Data Science & Machine Learning
- Web Development
- DevOps & Infrastructure
- Mobile App Development
- Security & Cryptography
- Game Development

## Quick Start

### Prerequisites

- Docker and Docker Compose (optional)
- A Base network wallet address (for payment functionality, optional for testing)

### Build and Run

```bash
# Build the Docker image
docker build -t repo-classifier .

# Run the container
docker run -p 8000:8000 -e MY_WALLET_ADDRESS=your_wallet_address repo-classifier
```

**Note**: The model downloads on first run, which may take a few minutes. The app will still function without x402 (rate limiting works, but payment enforcement is disabled).

### Test the API

```bash
# Health check
curl http://localhost:8000/

# Classify a repository (GET endpoint)
curl "http://localhost:8000/classify?repo_url=https://github.com/pandas-dev/pandas"

# Classify a repository (POST endpoint)
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/pandas-dev/pandas"}'
```

## API Documentation

Once the server is running, interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### `GET /`
Health check endpoint (always free, no payment required).

**Response:**
```json
{
  "message": "Repository Classifier API",
  "status": "ready",
  "pricing": {
    "free_tier": "5 requests per minute",
    "paid_tier": "$0.01 USD per request (via x402)",
    "note": "First 5 requests per minute are free. Beyond that, payment required."
  },
  "docs": "/docs",
  "example": "/classify?repo_url=https://github.com/pandas-dev/pandas"
}
```

#### `GET /classify?repo_url=<url>`
Classify a repository via GET request (browser-friendly).

**Parameters:**
- `repo_url` (query string): GitHub repository URL

#### `POST /classify`
Classify a repository via POST request.

**Request Body:**
```json
{
  "repo_url": "https://github.com/pandas-dev/pandas"
}
```

**Response:**
```json
{
  "status": "Free Tier",
  "category": "Data Science & Machine Learning",
  "confidence": 0.40614697337150574,
  "all_scores": {
    "Data Science & Machine Learning": 0.40614697337150574,
    "DevOps & Infrastructure": 0.16924525797367096,
    "Mobile App Development": 0.15506191551685333,
    "Web Development": 0.149778351187706,
    "Game Development": 0.07861993461847305,
    "Security & Cryptography": 0.04114754870533943
  },
  "repo_context": "Flexible and powerful data analysis / manipulation library for Python...",
  "rate_limit_remaining": 4
}
```

### Rate Limit Headers

All `/classify` responses include rate limit information:

- `X-RateLimit-Limit`: Maximum requests per minute (5)
- `X-RateLimit-Remaining`: Remaining free requests in current window
- `X-RateLimit-Status`: `free_tier` or `paid_required`

### Payment (x402)

When rate limit is exceeded, the API returns HTTP 402 with payment requirements:

```json
{
  "x402Version": 1,
  "accepts": [
    {
      "scheme": "exact",
      "network": "base",
      "maxAmountRequired": "10000",
      "resource": "http://localhost:8000/classify",
      "description": "Repository classification service",
      "payTo": "0x...",
      "maxTimeoutSeconds": 60
    }
  ]
}
```

Include the `X-PAYMENT` header with a valid payment token to proceed.

## Configuration

### Environment Variables

- `MY_WALLET_ADDRESS` (optional): Base network wallet address to receive payments. If not set, payment functionality is disabled but rate limiting still works.

### Rate Limiting

- **Free Tier**: 5 requests per minute per IP address
- **Window**: 60-second sliding window
- **Tracking**: In-memory (resets on container restart)

## Pricing Model

1. **Free Tier**: First 5 requests per minute are free
2. **Paid Tier**: $0.01 USD per request via x402 protocol on Base network

Rate limits are tracked per IP address using a sliding window algorithm.

## Examples

### Test Free Tier

```bash
# Make 5 requests (all free)
for i in {1..5}; do
  curl -X POST "http://localhost:8000/classify" \
    -H "Content-Type: application/json" \
    -d '{"repo_url": "https://github.com/pandas-dev/pandas"}' \
    | jq '{status, category, confidence, rate_limit_remaining}'
  sleep 1
done
```

### Test Different Repository Types

```bash
# Data Science repo
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/pandas-dev/pandas"}'

# Web Development repo
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/vercel/next.js"}'

# DevOps repo
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/kubernetes/kubernetes"}'
```

### Check Rate Limit Headers

```bash
curl -v -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/pandas-dev/pandas"}' 2>&1 | grep -i "x-ratelimit"
```

### Test Error Handling

```bash
# Invalid URL
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "not-a-valid-url"}'

# Non-existent repository
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/nonexistent/user/repo"}'
```

## Testing

A test script is provided in `scripts/test.sh`:

```bash
chmod +x scripts/test.sh
./scripts/test.sh
```

This script tests:
- Health check endpoint
- Free tier requests (5 requests)
- Rate limiting (6th request requiring payment)
- Rate limit headers

## Development

### Project Structure

```
repo-classifier/
├── app.py                 # Main application file
├── categories.json        # Category definitions (customizable)
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker container definition
├── .dockerignore         # Docker ignore patterns
├── .gitignore           # Git ignore patterns
├── README.md            # This file
└── scripts/
    └── test.sh          # Test script with configurable repositories
```

### Customizing Categories

Edit `categories.json` to customize the classification categories. The file contains:
- `labels`: Full descriptive labels for the zero-shot model
- `short_names`: Short names for API responses

After modifying `categories.json`, rebuild the Docker image for changes to take effect.

### Customizing Test Repositories

Edit `scripts/test.sh` and modify the `TEST_REPOS` array to test different repositories:

```bash
TEST_REPOS=(
  "https://github.com/your-org/your-repo"
  "https://github.com/another-org/another-repo"
  # Add more repositories here
)
```

### Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
   Or using uvicorn directly:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

### Dependencies

- **FastAPI**: Modern web framework for building APIs
- **transformers**: Hugging Face transformers library
- **torch**: PyTorch for ML model inference
- **x402**: Payment protocol implementation
- **requests**: HTTP client for GitHub API
- **uvicorn**: ASGI server

## Architecture

The application uses:
- **Zero-shot classification** with DeBERTa-v3 model for categorization
- **Sliding window rate limiting** for free tier management
- **x402 protocol** for payment processing
- **FastAPI middleware** for rate limiting and payment settlement
- **In-memory rate limit tracking** (per-IP address)

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

