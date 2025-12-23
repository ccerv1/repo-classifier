"""
Repo Classifier RAG Agent - FastAPI Application

A RAG-based GitHub repository classifier with x402 payment gating.
"""

import asyncio
import base64
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from repo_classifier.analyzer import Analyzer, AnalysisResult
from repo_classifier.config import settings
from repo_classifier.github_client import GitHubClient, github_client
from repo_classifier.rag import get_rag_store

# ============================================================================
# x402 Payment Integration (optional)
# ============================================================================

try:
    from x402.common import process_price_to_atomic_amount, find_matching_payment_requirements
    from x402.encoding import safe_base64_decode
    from x402.facilitator import FacilitatorClient
    from x402.types import PaymentPayload, PaymentRequirements, SupportedNetworks
    from typing import cast
    import json as json_module
    X402_AVAILABLE = True
    print("✓ x402 payment module loaded", flush=True)
except ImportError as e:
    X402_AVAILABLE = False
    print(f"⚠ x402 not available ({e}). Payment enforcement disabled.", flush=True)


# ============================================================================
# Rate Limiting
# ============================================================================

rate_limit_store: dict[str, deque] = defaultdict(lambda: deque())


def get_client_ip(request: Request) -> str:
    """Extract client IP from request headers or connection."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    if request.client:
        return request.client.host
    
    return "unknown"


def check_rate_limit(ip_address: str) -> tuple[bool, int]:
    """
    Check if IP is within free tier limits.
    
    Returns:
        (is_free, remaining_requests)
    """
    if not settings.free_tier_enabled:
        return False, 0
    
    now = time.time()
    request_times = rate_limit_store[ip_address]
    
    # Remove old entries outside the window
    while request_times and (now - request_times[0]) > settings.rate_limit_window:
        request_times.popleft()
    
    current_count = len(request_times)
    remaining = max(0, settings.free_tier_requests - current_count)
    
    if current_count < settings.free_tier_requests:
        request_times.append(now)
        return True, remaining - 1
    
    return False, 0


# ============================================================================
# Middleware
# ============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to track rate limits and set request state."""
    
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/analyze":
            ip_address = get_client_ip(request)
            is_free, remaining = check_rate_limit(ip_address)
            
            request.state.is_free_tier = is_free
            request.state.client_ip = ip_address
            request.state.rate_limit_remaining = remaining
        else:
            request.state.is_free_tier = True
            request.state.rate_limit_remaining = None
        
        response = await call_next(request)
        
        # Add rate limit headers
        if request.url.path == "/analyze":
            response.headers["X-RateLimit-Limit"] = str(settings.free_tier_requests)
            response.headers["X-RateLimit-Remaining"] = str(
                getattr(request.state, "rate_limit_remaining", 0)
            )
            response.headers["X-RateLimit-Window"] = str(settings.rate_limit_window)
        
        return response


class PaymentSettlementMiddleware(BaseHTTPMiddleware):
    """Middleware to settle x402 payments after successful requests."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Settle payment if applicable
        if (
            X402_AVAILABLE
            and hasattr(request.state, "payment_details")
            and 200 <= response.status_code < 300
        ):
            try:
                facilitator = FacilitatorClient()
                settle_response = await facilitator.settle(
                    request.state.payment_payload,
                    request.state.payment_details,
                )
                if settle_response.success:
                    response.headers["X-Payment-Response"] = base64.b64encode(
                        settle_response.model_dump_json(by_alias=True).encode()
                    ).decode()
            except Exception as e:
                print(f"⚠ Payment settlement failed: {e}", flush=True)
        
        return response


# ============================================================================
# Request/Response Models
# ============================================================================

class AnalyzeRequest(BaseModel):
    """Request body for /analyze endpoint."""
    
    url: str = Field(
        ...,
        description="GitHub repository URL",
        examples=["https://github.com/astral-sh/uv"],
    )
    categories: dict[str, str] = Field(
        ...,
        description="Available categories with descriptions",
        examples=[{
            "DevOps": "Tools for infrastructure and CI/CD",
            "Web": "Frontend and backend frameworks",
            "System": "Low-level tooling and package managers",
        }],
    )
    
    @field_validator("url")
    @classmethod
    def validate_github_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("URL cannot be empty")
        if "github.com" not in v.lower():
            raise ValueError("URL must be a GitHub repository URL")
        return v
    
    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v: dict) -> dict:
        if not v:
            raise ValueError("At least one category is required")
        if len(v) > 20:
            raise ValueError("Maximum 20 categories allowed")
        return v


class AnalyzeResponseData(BaseModel):
    """Data payload in successful response."""
    category: str
    confidence: float
    reasoning: str
    similar_precedents: list[str]


class AnalyzeResponse(BaseModel):
    """Response body for /analyze endpoint."""
    status: str = "success"
    data: AnalyzeResponseData


class ErrorResponse(BaseModel):
    """Error response body."""
    error: str
    detail: str | None = None


class PaymentRequiredResponse(BaseModel):
    """402 Payment Required response."""
    error: str = "Payment Required"
    x402Version: int = 1
    accepts: list[dict[str, Any]]


# ============================================================================
# Application Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("🚀 Starting Repo Classifier RAG Agent...", flush=True)
    print(f"   Free tier: {settings.free_tier_requests} requests per {settings.rate_limit_window}s", flush=True)
    print(f"   Price: ${settings.price_per_request} per request", flush=True)
    print(f"   ChromaDB: {settings.chroma_db_path}", flush=True)
    
    # Initialize RAG store
    rag_store = get_rag_store()
    print(f"   RAG store: {rag_store.count()} repositories loaded", flush=True)
    
    yield
    
    # Cleanup
    await github_client.close()
    print("👋 Shutdown complete", flush=True)


app = FastAPI(
    title="Repo Classifier RAG Agent",
    description="RAG-based GitHub repository classifier with x402 payment gating",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(RateLimitMiddleware)
if X402_AVAILABLE and settings.wallet_address:
    app.add_middleware(PaymentSettlementMiddleware)


# ============================================================================
# Payment Handling
# ============================================================================

def get_payment_requirements(resource_url: str) -> list:
    """Build x402 payment requirements."""
    if not X402_AVAILABLE or not settings.wallet_address:
        return []
    
    try:
        network = "base"
        price_str = str(settings.price_per_request)
        
        max_amount, asset_address, eip712_domain = process_price_to_atomic_amount(
            price_str, network
        )
        
        return [
            PaymentRequirements(
                scheme="exact",
                network=cast(SupportedNetworks, network),
                asset=asset_address,
                max_amount_required=max_amount,
                resource=resource_url,
                description=f"Repository classification: ${settings.price_per_request} USD",
                mime_type="application/json",
                pay_to=settings.wallet_address,
                max_timeout_seconds=60,
                output_schema={
                    "input": {"type": "http", "method": "POST", "discoverable": True},
                    "output": None,
                },
                extra=eip712_domain,
            )
        ]
    except Exception as e:
        print(f"⚠ Failed to build payment requirements: {e}", flush=True)
        return []


async def verify_payment(request: Request) -> JSONResponse | None:
    """
    Verify x402 payment if required.
    
    Returns:
        JSONResponse with 402 if payment required/invalid, None if payment valid
    """
    if not X402_AVAILABLE or not settings.wallet_address:
        return None  # Payment not configured
    
    resource_url = str(request.url)
    payment_reqs = get_payment_requirements(resource_url)
    
    if not payment_reqs:
        return None
    
    # Check for payment header
    payment_header = request.headers.get("X-PAYMENT", "")
    
    if not payment_header:
        # Return 402 with payment requirements
        return JSONResponse(
            status_code=402,
            content={
                "error": "Payment Required",
                "x402Version": 1,
                "accepts": [req.model_dump(by_alias=True) for req in payment_reqs],
            },
        )
    
    # Verify payment
    try:
        payment_dict = json_module.loads(safe_base64_decode(payment_header))
        payment = PaymentPayload(**payment_dict)
        
        selected_req = find_matching_payment_requirements(payment_reqs, payment)
        if not selected_req:
            return JSONResponse(
                status_code=402,
                content={"error": "No matching payment requirements"},
            )
        
        facilitator = FacilitatorClient()
        verify_result = await facilitator.verify(payment, selected_req)
        
        if not verify_result.is_valid:
            return JSONResponse(
                status_code=402,
                content={
                    "error": "Invalid payment",
                    "detail": verify_result.invalid_reason,
                },
            )
        
        # Store for settlement
        request.state.payment_details = selected_req
        request.state.payment_payload = payment
        
        return None  # Payment valid
        
    except Exception as e:
        return JSONResponse(
            status_code=402,
            content={"error": "Payment verification failed", "detail": str(e)},
        )


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check and API info."""
    rag_store = get_rag_store()
    
    return {
        "service": "Repo Classifier RAG Agent",
        "version": "1.0.0",
        "status": "ready",
        "rag_store": {
            "repositories": rag_store.count(),
            "categories": rag_store.list_categories(),
        },
        "pricing": {
            "free_tier": f"{settings.free_tier_requests} requests per {settings.rate_limit_window}s" if settings.free_tier_enabled else "disabled",
            "paid": f"${settings.price_per_request} USD per request",
            "payment_enabled": X402_AVAILABLE and bool(settings.wallet_address),
        },
        "docs": "/docs",
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repository(body: AnalyzeRequest, request: Request):
    """
    Analyze a GitHub repository and classify it.
    
    Uses RAG to find similar repositories and LLM to synthesize a classification.
    
    **Pricing:**
    - Free tier: First N requests per minute (configurable)
    - Paid: $0.02 per request via x402 protocol
    """
    # Check if payment required (beyond free tier)
    is_free_tier = getattr(request.state, "is_free_tier", False)
    
    if not is_free_tier:
        payment_response = await verify_payment(request)
        if payment_response:
            return payment_response
    
    try:
        # Parse GitHub URL
        github = GitHubClient()
        owner, repo = github.parse_github_url(body.url)
        
        # Get RAG store
        rag_store = get_rag_store()
        
        # Parallel: investigate repo and query RAG
        evidence, precedents = await asyncio.gather(
            github.investigate_repo(owner, repo),
            rag_store.query_similar(
                readme="",  # Will be populated after investigation
                description=None,
            ),
        )
        
        # Now query RAG with actual evidence
        precedents = await rag_store.query_similar(
            readme=evidence.readme,
            description=evidence.description,
        )
        
        # Analyze with LLM
        analyzer = Analyzer()
        result: AnalysisResult = await analyzer.analyze(
            evidence=evidence,
            precedents=precedents,
            categories=body.categories,
        )
        
        return AnalyzeResponse(
            status="success",
            data=AnalyzeResponseData(
                category=result.category,
                confidence=result.confidence,
                reasoning=result.reasoning,
                similar_precedents=result.similar_precedents,
            ),
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Analysis error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Simple health check for container orchestration."""
    return {"status": "healthy"}


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    print(f"❌ Unhandled error: {exc}", flush=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


# ============================================================================
# Main
# ============================================================================

def main():
    """Entry point for running the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

