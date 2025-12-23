#!/usr/bin/env python3
"""
Taxonomy dataset ingestion script.

Parses a JSON file of repositories with known categories and populates
the ChromaDB vector store for RAG-based classification.

Usage:
    uv run repo-classifier-ingest --input data/taxonomy.json
    uv run repo-classifier-ingest --input data/taxonomy.json --clear
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from repo_classifier.config import settings
from repo_classifier.github_client import GitHubClient
from repo_classifier.rag import RAGStore


async def ingest_repositories(
    input_file: Path,
    clear: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Ingest repositories from a JSON file into the RAG store.
    
    Args:
        input_file: Path to JSON file with repository data
        clear: If True, clear existing data before ingesting
        verbose: If True, print progress messages
        
    Returns:
        Dict with ingestion statistics
    """
    # Load input file
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file) as f:
        data = json.load(f)
    
    repositories = data.get("repositories", [])
    
    if not repositories:
        raise ValueError("No repositories found in input file")
    
    if verbose:
        print(f"Found {len(repositories)} repositories to ingest")
        print(f"ChromaDB path: {settings.chroma_db_path}")
    
    # Initialize stores
    rag_store = RAGStore()
    github = GitHubClient()
    
    try:
        # Clear if requested
        if clear:
            if verbose:
                print("Clearing existing data...")
            rag_store.clear()
        
        # Track statistics
        stats = {
            "total": len(repositories),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
        }
        
        # Process each repository
        for i, repo in enumerate(repositories, 1):
            url = repo.get("url")
            category = repo.get("category")
            
            if not url or not category:
                if verbose:
                    print(f"  [{i}/{len(repositories)}] Skipping: missing url or category")
                stats["skipped"] += 1
                continue
            
            try:
                if verbose:
                    print(f"  [{i}/{len(repositories)}] Processing: {url}")
                
                # Parse GitHub URL
                owner, repo_name = GitHubClient.parse_github_url(url)
                
                # Fetch repository data
                evidence = await github.investigate_repo(owner, repo_name)
                
                # Add to RAG store
                await rag_store.add_repository(
                    url=url,
                    readme=evidence.readme,
                    category=category,
                    description=evidence.description,
                )
                
                stats["success"] += 1
                
                if verbose:
                    print(f"           Category: {category} | README: {len(evidence.readme)} chars")
                
            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append({"url": url, "error": str(e)})
                
                if verbose:
                    print(f"           ERROR: {e}")
        
        # Print summary
        if verbose:
            print("\n" + "=" * 50)
            print("Ingestion Complete")
            print("=" * 50)
            print(f"  Total:     {stats['total']}")
            print(f"  Success:   {stats['success']}")
            print(f"  Failed:    {stats['failed']}")
            print(f"  Skipped:   {stats['skipped']}")
            print(f"\nRAG store now contains {rag_store.count()} repositories")
            
            # Show category distribution
            categories = rag_store.list_categories()
            if categories:
                print("\nCategories:")
                for cat, count in sorted(categories.items()):
                    print(f"  - {cat}: {count}")
        
        return stats
        
    finally:
        await github.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest taxonomy repositories into RAG store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run repo-classifier-ingest --input data/taxonomy.json
  uv run repo-classifier-ingest --input data/taxonomy.json --clear
  uv run repo-classifier-ingest --input data/taxonomy.json --quiet

Expected JSON format:
{
  "repositories": [
    {"url": "https://github.com/owner/repo", "category": "Web"},
    ...
  ]
}
        """,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to JSON file with repository data",
    )
    
    parser.add_argument(
        "--clear", "-c",
        action="store_true",
        help="Clear existing data before ingesting",
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    try:
        stats = asyncio.run(ingest_repositories(
            input_file=args.input,
            clear=args.clear,
            verbose=not args.quiet,
        ))
        
        # Exit with error if any failures
        if stats["failed"] > 0:
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

