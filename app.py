# app.py
import os
import base64
import re
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple
from urllib.parse import urlparse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import requests
from transformers import pipeline

# Import x402 payment functionality
try:
    from x402.common import process_price_to_atomic_amount, find_matching_payment_requirements
    from x402.encoding import safe_base64_decode
    from x402.facilitator import FacilitatorClient
    from x402.types import PaymentPayload, PaymentRequirements, SupportedNetworks
    from x402.fastapi.middleware import is_browser_request, get_paywall_html
    from typing import cast
    import json
    X402_AVAILABLE = True
    print("✓ Successfully imported x402 payment functionality", flush=True)
except ImportError as e:
    X402_AVAILABLE = False
    print(f"WARNING: x402 not found ({e}). Payment functionality disabled.", flush=True)
    print("  Rate limiting will still work, but payment enforcement is disabled.", flush=True)

# Load category definitions from JSON file
def load_categories() -> Tuple[List[str], Dict[str, str]]:
    """Load category labels and mappings from categories.json file."""
    categories_file = os.path.join(os.path.dirname(__file__), "categories.json")
    try:
        with open(categories_file, "r") as f:
            data = json.load(f)
            categories = data.get("categories", {})
            labels = categories.get("labels", [])
            short_names = categories.get("short_names", {})
            return labels, short_names
    except FileNotFoundError:
        print(f"WARNING: categories.json not found at {categories_file}. Using defaults.", flush=True)
        # Fallback to defaults
        return _default_categories()
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in categories.json: {e}. Using defaults.", flush=True)
        return _default_categories()

def _default_categories() -> Tuple[List[str], Dict[str, str]]:
    """Default category definitions as fallback."""
    labels = [
        "This is a data science or machine learning project involving data analysis, statistics, AI models, or scientific computing",
        "This is a web development project for building websites, web applications, or web services",
        "This is a DevOps or infrastructure project for deployment, CI/CD, containerization, or system administration",
        "This is a mobile app development project for iOS, Android, or cross-platform mobile applications",
        "This is a security or cryptography project for encryption, authentication, vulnerability analysis, or security tools",
        "This is a game development project for creating video games or game engines"
    ]
    short_names = {
        labels[0]: "Data Science & Machine Learning",
        labels[1]: "Web Development",
        labels[2]: "DevOps & Infrastructure",
        labels[3]: "Mobile App Development",
        labels[4]: "Security & Cryptography",
        labels[5]: "Game Development"
    }
    return labels, short_names

CATEGORY_LABELS, CATEGORY_MAP = load_categories()

# Rate limiting configuration
FREE_TIER_REQUESTS_PER_MINUTE = 5
RATE_LIMIT_WINDOW = 60  # 60 seconds

# In-memory rate limit tracking (sliding window per IP)
# Format: {ip_address: deque([timestamp1, timestamp2, ...])}
rate_limit_store: Dict[str, deque] = defaultdict(lambda: deque())

def get_client_ip(request: Request) -> str:
    """Extract client IP address from request."""
    # Check for forwarded IP (if behind proxy/load balancer)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # X-Forwarded-For can contain multiple IPs, take the first one
        return forwarded.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fallback to direct client IP
    if request.client:
        return request.client.host
    return "unknown"

def is_within_free_tier(ip_address: str) -> tuple[bool, int]:
    """
    Check if IP address is within free tier (5 requests per minute).
    Returns (is_free, remaining_count) where remaining_count is BEFORE this request.
    """
    now = time.time()
    
    # Get request history for this IP
    request_times = rate_limit_store[ip_address]
    
    # Remove timestamps older than 1 minute (sliding window)
    while request_times and (now - request_times[0]) > RATE_LIMIT_WINDOW:
        request_times.popleft()
    
    # Calculate remaining BEFORE adding this request
    current_count = len(request_times)
    remaining = max(0, FREE_TIER_REQUESTS_PER_MINUTE - current_count)
    
    # Check if under limit (before adding this request)
    if current_count < FREE_TIER_REQUESTS_PER_MINUTE:
        # Record this request
        request_times.append(now)
        return True, remaining - 1  # Remaining after this request
    
    # Over limit
    return False, 0

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to track rate limits and set free tier flag in request state."""
    
    async def dispatch(self, request: Request, call_next):
        # Only apply rate limiting to /classify endpoints
        if request.url.path.startswith("/classify"):
            ip_address = get_client_ip(request)
            is_free, remaining = is_within_free_tier(ip_address)
            
            # Store in request state for use in x402 or endpoints
            request.state.rate_limit_free_tier = is_free
            request.state.client_ip = ip_address
            request.state.rate_limit_remaining = remaining
        else:
            # For other endpoints, set default values
            request.state.rate_limit_free_tier = True
            request.state.rate_limit_remaining = None
        
        response = await call_next(request)
        
        # Add rate limit headers to response
        if request.url.path.startswith("/classify"):
            response.headers["X-RateLimit-Limit"] = str(FREE_TIER_REQUESTS_PER_MINUTE)
            response.headers["X-RateLimit-Remaining"] = str(getattr(request.state, 'rate_limit_remaining', 0))
            if not getattr(request.state, 'rate_limit_free_tier', False):
                response.headers["X-RateLimit-Status"] = "paid_required"
            else:
                response.headers["X-RateLimit-Status"] = "free_tier"
        
        return response

# Initialize FastAPI app
app = FastAPI(title="Paid Repo Classifier", version="1.0.0")

# Load the classifier model once at startup (stays in memory)
# Using DeBERTa-v3 which is state-of-the-art for zero-shot classification
print("Loading classifier model (this may take a moment on first run)...", flush=True)
try:
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    print("Model loaded successfully!", flush=True)
except Exception as e:
    print(f"ERROR: Failed to load model: {e}", flush=True)
    raise

# Initialize x402 payment configuration
X402_WALLET_ADDRESS = None
X402_FACILITATOR = None
X402_PAYMENT_REQUIREMENTS = None

if X402_AVAILABLE:
    wallet_address = os.getenv("MY_WALLET_ADDRESS")
    if not wallet_address:
        print("WARNING: MY_WALLET_ADDRESS not set. x402 payment disabled.", flush=True)
        X402_AVAILABLE = False
    else:
        X402_WALLET_ADDRESS = wallet_address
        X402_FACILITATOR = FacilitatorClient()
        
        # Create payment requirements for /classify endpoint
        try:
            network = "base"  # Base mainnet
            price_str = "0.01"  # $0.01 USD
            
            max_amount_required, asset_address, eip712_domain = process_price_to_atomic_amount(
                price_str, network
            )
            
            X402_PAYMENT_REQUIREMENTS = [
                PaymentRequirements(
                    scheme="exact",
                    network=cast(SupportedNetworks, network),
                    asset=asset_address,
                    max_amount_required=max_amount_required,
                    resource=None,  # Will be set per request
                    description="Repository classification service",
                    mime_type="application/json",
                    pay_to=wallet_address,
                    max_timeout_seconds=60,
                    output_schema={
                        "input": {
                            "type": "http",
                            "method": "POST",
                            "discoverable": True,
                        },
                        "output": None,
                    },
                    extra=eip712_domain,
                )
            ]
            print("✓ x402 payment configured: $0.01 per request (beyond free tier)", flush=True)
        except Exception as e:
            print(f"WARNING: Failed to configure x402 payment requirements: {e}", flush=True)
            X402_AVAILABLE = False
else:
    print("x402 not available - running without payment gateway.", flush=True)

# Add rate limiting middleware
# This tracks requests and sets request.state.rate_limit_free_tier flag
app.add_middleware(RateLimitMiddleware)

# Add response middleware to settle x402 payments after successful requests
class PaymentSettlementMiddleware(BaseHTTPMiddleware):
    """Middleware to settle x402 payments after successful request processing."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Settle payment if payment was verified and request was successful
        if X402_AVAILABLE and hasattr(request.state, 'payment_details') and response.status_code >= 200 and response.status_code < 300:
            try:
                settle_response = await X402_FACILITATOR.settle(
                    request.state.payment_payload,
                    request.state.payment_details
                )
                if settle_response.success:
                    response.headers["X-PAYMENT-RESPONSE"] = base64.b64encode(
                        settle_response.model_dump_json(by_alias=True).encode("utf-8")
                    ).decode("utf-8")
            except Exception as e:
                # Log error but don't fail the response (payment already verified)
                print(f"Warning: Payment settlement failed: {e}", flush=True)
        
        return response

if X402_AVAILABLE:
    app.add_middleware(PaymentSettlementMiddleware)

print("Rate limiting enabled. API server ready.", flush=True)

# Request/Response models
class ClassifyRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL (e.g., https://github.com/owner/repo)")
    
    @field_validator('repo_url')
    @classmethod
    def validate_repo_url(cls, v: str) -> str:
        """Validate that the URL is a valid GitHub repository URL."""
        v = v.strip()
        if not v:
            raise ValueError("Repository URL cannot be empty")
        
        # Check if it's a valid URL
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL format")
        
        # Check if it's a GitHub URL
        if 'github.com' not in parsed.netloc:
            raise ValueError("URL must be a GitHub repository URL")
        
        # Extract owner/repo from path
        path_parts = [p for p in parsed.path.strip('/').split('/') if p]
        if len(path_parts) < 2:
            raise ValueError("GitHub URL must include owner and repository name (e.g., https://github.com/owner/repo)")
        
        return v

class ClassifyResponse(BaseModel):
    status: str  # "Free Tier" or "Paid & Delivered"
    category: str
    confidence: float
    all_scores: Dict[str, float]
    repo_context: str
    rate_limit_remaining: int = None

def get_repo_data(repo_url: str) -> str:
    """
    Fetches repository metadata and README from GitHub API.
    Returns combined text for classification.
    
    Args:
        repo_url: GitHub repository URL
        
    Returns:
        Combined text for classification
        
    Raises:
        HTTPException: If repository is not found or API call fails
        ValueError: If URL format is invalid
    """
    # Parse owner/repo from URL (more robust parsing)
    parsed = urlparse(repo_url.strip())
    path_parts = [p for p in parsed.path.strip('/').split('/') if p]
    
    if len(path_parts) < 2:
        raise ValueError("Invalid GitHub URL format: must include owner and repository")
    
    owner = path_parts[-2]
    repo = path_parts[-1]
    
    # Remove .git suffix if present
    repo = repo.rstrip('.git')
    
    # Fetch repository metadata
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad responses
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="GitHub API request timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to GitHub API: {str(e)}")
    
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail=f"Repository not found: {repo_url}")
    
    data = response.json()
    description = data.get("description", "") or ""
    repo_name = data.get("name", repo)
    
    # Fetch README content
    readme_content = ""
    try:
        readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        readme_response = requests.get(readme_url, timeout=10)
        if readme_response.status_code == 200:
            readme_data = readme_response.json()
            # Decode base64 content
            readme_encoded = readme_data.get("content", "")
            if readme_encoded:
                readme_content = base64.b64decode(readme_encoded).decode('utf-8', errors='ignore')
                # Clean up markdown and extract key information (first 1500 chars for better context)
                # Remove markdown headers, links, code blocks for cleaner text
                readme_content = re.sub(r'#{1,6}\s+', '', readme_content)  # Remove headers
                readme_content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', readme_content)  # Remove links, keep text
                readme_content = re.sub(r'```[\s\S]*?```', '', readme_content)  # Remove code blocks
                readme_content = re.sub(r'`([^`]+)`', r'\1', readme_content)  # Remove inline code
                readme_content = ' '.join(readme_content.split())  # Normalize whitespace
                readme_content = readme_content[:1500]  # First 1500 chars
    except Exception as e:
        # If README fetch fails, continue without it
        print(f"Warning: Could not fetch README: {e}", flush=True)
    
    # Build better text for classification - prioritize description, then README intro
    # Focus on what the project DOES rather than just the name
    text_parts = []
    if description:
        text_parts.append(description)
    if readme_content:
        # Use the first portion of README (usually contains the key info)
        text_parts.append(readme_content)
    
    # If we have no description or README, use the repo name
    if not text_parts:
        text_parts.append(f"GitHub repository: {repo_name}")
    
    text_to_analyze = " ".join(text_parts)
    return text_to_analyze

@app.get("/")
def root():
    """Health check endpoint (free, no payment required)"""
    return {
        "message": "Repository Classifier API", 
        "status": "ready",
        "pricing": {
            "free_tier": f"{FREE_TIER_REQUESTS_PER_MINUTE} requests per minute",
            "paid_tier": "$0.01 USD per request (via x402)",
            "note": "First 5 requests per minute are free. Beyond that, payment required."
        },
        "docs": "/docs",
        "example": "/classify?repo_url=https://github.com/pandas-dev/pandas"
    }

@app.get("/classify", response_model=ClassifyResponse)
async def classify_repo_get(repo_url: str, request: Request):
    """
    Classify a GitHub repository (GET endpoint for browser testing).
    Example: /classify?repo_url=https://github.com/pandas-dev/pandas
    
    Free tier: 5 requests per minute. Beyond that, payment via x402 is required.
    """
    # Check payment if beyond free tier
    is_free_tier = getattr(request.state, 'rate_limit_free_tier', False)
    if not is_free_tier and X402_AVAILABLE:
        payment_response = await check_x402_payment(request)
        if payment_response:  # Returns JSONResponse if payment required
            return payment_response
    
    return _classify_repo_internal(repo_url, request)

@app.post("/classify", response_model=ClassifyResponse)
async def classify_repo(api_request: ClassifyRequest, request: Request):
    """
    Classify a GitHub repository into predefined categories (POST endpoint).
    
    Free tier: 5 requests per minute. Beyond that, payment via x402 is required.
    """
    # Check payment if beyond free tier
    is_free_tier = getattr(request.state, 'rate_limit_free_tier', False)
    if not is_free_tier and X402_AVAILABLE:
        payment_response = await check_x402_payment(request)
        if payment_response:  # Returns JSONResponse if payment required
            return payment_response
    
    return _classify_repo_internal(api_request.repo_url, request)

async def check_x402_payment(request: Request):
    """
    Check x402 payment if request is beyond free tier.
    Raises HTTPException(402) if payment is required but invalid/missing.
    """
    if not X402_AVAILABLE or not X402_PAYMENT_REQUIREMENTS:
        return  # x402 not configured, skip payment check
    
    # Update resource URL in payment requirements (create new instances)
    resource_url = str(request.url)
    payment_reqs = []
    for req in X402_PAYMENT_REQUIREMENTS:
        # Create a new PaymentRequirements with updated resource URL
        req_dict = req.model_dump()
        req_dict['resource'] = resource_url
        payment_reqs.append(PaymentRequirements(**req_dict))
    
    # Check for payment header
    payment_header = request.headers.get("X-PAYMENT", "")
    
    if payment_header == "":
        # Return 402 with payment requirements
        request_headers = dict(request.headers)
        
        if is_browser_request(request_headers):
            html_content = get_paywall_html(
                payment_reqs,
                request.url.path,
            )
            raise HTTPException(
                status_code=402,
                detail="Payment required"
            )
        else:
            # Return proper 402 JSON response with payment requirements
            response_data = {
                "x402Version": 1,
                "accepts": [req.model_dump(by_alias=True) for req in payment_reqs],
            }
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=402,
                content=response_data
            )
    
    # Decode and validate payment header
    try:
        payment_dict = json.loads(safe_base64_decode(payment_header))
        payment = PaymentPayload(**payment_dict)
    except Exception as e:
        raise HTTPException(
            status_code=402,
            detail={"error": "Invalid payment header format", "details": str(e)}
        )
    
    # Find matching payment requirements
    selected_payment_requirements = find_matching_payment_requirements(
        payment_reqs, payment
    )
    
    if not selected_payment_requirements:
        raise HTTPException(
            status_code=402,
            detail={"error": "No matching payment requirements found"}
        )
    
    # Verify payment
    verify_response = await X402_FACILITATOR.verify(
        payment, selected_payment_requirements
    )
    
    if not verify_response.is_valid:
        error_reason = verify_response.invalid_reason or "Unknown error"
        raise HTTPException(
            status_code=402,
            detail={"error": "Invalid payment", "reason": error_reason}
        )
    
    # Store payment details in request state for settlement after processing
    request.state.payment_details = selected_payment_requirements
    request.state.payment_payload = payment

def _classify_repo_internal(repo_url: str, request: Request) -> ClassifyResponse:
    """
    Internal function to classify a repository.
    
    Args:
        repo_url: GitHub repository URL
        request: FastAPI request object (for accessing rate limit state)
        
    Returns:
        ClassifyResponse with classification results
        
    Raises:
        HTTPException: If repository cannot be fetched or classified
    """
    # Check if this request is in free tier
    is_free_tier = getattr(request.state, 'rate_limit_free_tier', False)
    remaining = getattr(request.state, 'rate_limit_remaining', 0)
    # Fetch repository data (including README)
    try:
        context = get_repo_data(repo_url)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid repository URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
    # Classify using the predefined categories
    try:
        result = classifier(context, CATEGORY_LABELS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
    
    # Map back to short category names for cleaner output
    short_labels = [CATEGORY_MAP[label] for label in result['labels']]
    all_scores = dict(zip(short_labels, [float(score) for score in result['scores']]))
    top_category = CATEGORY_MAP[result['labels'][0]]
    
    # Determine status based on whether this was a free tier or paid request
    status = "Free Tier" if is_free_tier else "Paid & Delivered"
    
    return ClassifyResponse(
        status=status,
        category=top_category,
        confidence=float(result['scores'][0]),
        all_scores=all_scores,
        repo_context=context[:200] + "..." if len(context) > 200 else context,
        rate_limit_remaining=remaining if is_free_tier else None
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
