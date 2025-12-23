# RepoRank RAG Agent

A RAG-based GitHub repository classifier designed for the machine-to-machine economy. Autonomous agents can submit GitHub repository URLs and receive semantic categorization powered by historical precedents and LLM synthesis.

## Features

- **RAG Architecture**: Uses ChromaDB vector store with OpenAI embeddings to find similar repositories from a taxonomy dataset
- **OpenAI Structured Outputs**: Guaranteed valid JSON responses using `response_format: json_schema`
- **Async & Parallel**: Concurrent GitHub API calls and caching for optimal latency (<6s target)
- **Flexible Categories**: User-defined categories per request
- **x402 Payment Gating**: Optional payment enforcement via x402 protocol on Base network
- **Freemium Model**: Configurable free tier with rate limiting

## Project Structure

```
repo-classifier/
├── src/
│   └── repo_classifier/
│       ├── __init__.py
│       ├── app.py           # FastAPI application
│       ├── config.py        # Environment configuration
│       ├── cache.py         # TTL-based in-memory cache
│       ├── github_client.py # Async GitHub REST API client
│       ├── rag.py           # ChromaDB vector operations
│       ├── analyzer.py      # OpenAI LLM synthesis
│       └── ingest.py        # Taxonomy ingestion CLI
├── data/
│   └── taxonomy.json   # Your seed dataset
├── scripts/
│   └── test.sh              # API test script
├── pyproject.toml           # Dependencies (uv)
├── Dockerfile               # Production container
├── docker-compose.yml       # Local development
└── env.example              # Environment template
```

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- OpenAI API key
- GitHub token (optional, for higher rate limits)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/repo-classifier.git
cd repo-classifier

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Create environment file
cp env.example .env
# Edit .env with your API keys
```

### Running Locally

```bash
# Start the server
uv run repo-classifier

# In another terminal, ingest the taxonomy dataset
uv run repo-classifier-ingest --input data/taxonomy.json
```

### Using Docker

```bash
# Build and run
docker-compose up --build

# Ingest taxonomy data (first time only)
docker-compose exec reporank uv run repo-classifier-ingest --input data/taxonomy.json
```

## API Usage

### Analyze a Repository

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://github.com/solana-labs/solana",
    "categories": {
      "BNB Chain": "Binance Smart Chain and BNB ecosystem projects",
      "Deno": "Deno runtime, TypeScript/JavaScript server-side projects",
      "Ethereum": "Ethereum blockchain, EVM, smart contracts, and DeFi",
      "Hardhat": "Hardhat development framework and tooling projects",
      "Polygon": "Polygon network and Layer 2 scaling solutions",
      "Rust": "Rust programming language libraries and tools",
      "Solana": "Solana blockchain, programs, and ecosystem tools"
    }
  }'
```

### Response

```json
{
  "status": "success",
  "data": {
    "category": "Solana",
    "confidence": 0.95,
    "reasoning": "Repository is the core Solana blockchain implementation...",
    "similar_precedents": [
      "https://github.com/solana-developers/CRUD-dApp",
      "https://github.com/jacobcreech/wallet-adapter"
    ]
  }
}
```

### Health Check

```bash
curl http://localhost:8000/
```

## Configuration

All configuration is via environment variables. See `env.example` for the full list.

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `GITHUB_TOKEN` | GitHub token for higher rate limits | - |
| `WALLET_ADDRESS` | x402 payment recipient address | - |
| `FREE_TIER_ENABLED` | Enable free tier | `true` |
| `FREE_TIER_REQUESTS` | Free requests per window | `5` |
| `PRICE_PER_REQUEST` | USD price per request | `0.02` |
| `CHROMA_DB_PATH` | ChromaDB storage path | `./data/chroma_db` |

## Deployment

### Railway

1. Connect GitHub repo to Railway
2. Set environment variables in dashboard
3. Add volume: Mount path `/app/data/chroma_db`, Size 1GB
4. Ingest data via Railway shell:
   ```bash
   uv run repo-classifier-ingest --input data/taxonomy.json
   ```

### Render

1. Create Web Service, select Docker environment
2. Set environment variables
3. Add Persistent Disk: Mount path `/app/data/chroma_db`
4. Deploy and ingest via Render Shell

## Taxonomy Dataset

The RAG system requires a taxonomy dataset of repositories with known categories. Edit `data/taxonomy.json`:

```json
{
  "repositories": [
    {"url": "https://github.com/solana-labs/solana", "category": "Solana"},
    {"url": "https://github.com/denoland/deno", "category": "Deno"},
    {"url": "https://github.com/foundry-rs/foundry", "category": "Ethereum"}
  ]
}
```

Ingest after changes:
```bash
uv run repo-classifier-ingest --input data/taxonomy.json --clear
```

## CLI Commands

```bash
# Start the API server
uv run repo-classifier

# Ingest taxonomy data
uv run repo-classifier-ingest --input data/taxonomy.json

# Clear and re-ingest
uv run repo-classifier-ingest --input data/taxonomy.json --clear

# Quiet mode (no progress output)
uv run repo-classifier-ingest --input data/taxonomy.json --quiet
```

## Testing

```bash
# Run test script
chmod +x scripts/test.sh
./scripts/test.sh

# Or test manually
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/denoland/deno", "categories": {"Deno": "Deno runtime", "Rust": "Rust projects"}}'
```

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────────┐
│   Client    │────▶│  FastAPI + x402 Middleware                       │
│  (AI Agent) │     │                                                  │
└─────────────┘     │  ┌──────────────────┐  ┌─────────────────────┐  │
                    │  │ GitHub REST API  │  │ ChromaDB (RAG)      │  │
                    │  │ - File tree      │  │ - Taxonomy          │  │
                    │  │ - README         │  │ - k=3 neighbors     │  │
                    │  │ - Parallel fetch │  │ - Embeddings        │  │
                    │  └────────┬─────────┘  └──────────┬──────────┘  │
                    │           │                       │              │
                    │           ▼                       ▼              │
                    │  ┌────────────────────────────────────────────┐  │
                    │  │         OpenAI gpt-4o-mini                 │  │
                    │  │   (Structured Output: json_schema)         │  │
                    │  └────────────────────────────────────────────┘  │
                    └──────────────────────────────────────────────────┘
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Latency | < 6 seconds |
| Cost per request | < $0.005 |
| Accuracy | > 92% |

## License

MIT
