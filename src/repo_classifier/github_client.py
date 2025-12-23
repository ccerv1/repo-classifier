"""
Async GitHub REST API client for repository investigation.
Fetches file trees and README content with caching and parallel requests.
"""

import asyncio
import base64
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from repo_classifier.cache import github_cache
from repo_classifier.config import settings


@dataclass
class RepoEvidence:
    """Evidence collected from a GitHub repository."""
    owner: str
    repo: str
    file_tree: list[str]
    readme: str
    description: str | None = None


class GitHubClient:
    """
    Async client for GitHub REST API.
    
    Features:
    - Parallel fetching of file tree and README
    - Automatic caching with configurable TTL
    - Rate limit aware (uses token if available)
    """

    BASE_URL = "https://api.github.com"

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    @property
    def headers(self) -> dict[str, str]:
        """Build request headers with optional auth token."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Repo Classifier-Agent/1.0",
        }
        if settings.github_token:
            headers["Authorization"] = f"Bearer {settings.github_token}"
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=self.headers,
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @staticmethod
    def parse_github_url(url: str) -> tuple[str, str]:
        """
        Parse a GitHub URL to extract owner and repo.
        
        Args:
            url: GitHub repository URL
            
        Returns:
            Tuple of (owner, repo)
            
        Raises:
            ValueError: If URL is not a valid GitHub repository URL
        """
        url = url.strip()
        parsed = urlparse(url)

        if "github.com" not in parsed.netloc:
            raise ValueError("URL must be a GitHub repository URL")

        path_parts = [p for p in parsed.path.strip("/").split("/") if p]

        if len(path_parts) < 2:
            raise ValueError(
                "GitHub URL must include owner and repository name "
                "(e.g., https://github.com/owner/repo)"
            )

        owner = path_parts[0]
        repo = path_parts[1].removesuffix(".git")

        return owner, repo

    async def get_repo_info(self, owner: str, repo: str) -> dict:
        """
        Fetch repository metadata.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Repository metadata dict
        """
        cache_key = f"github:{owner}/{repo}:info"
        cached = github_cache.get(cache_key)
        if cached is not None:
            return cached

        client = await self._get_client()
        url = f"{self.BASE_URL}/repos/{owner}/{repo}"

        response = await client.get(url)
        response.raise_for_status()

        data = response.json()
        github_cache.set(cache_key, data, settings.cache_ttl_seconds)

        return data

    async def get_file_tree(self, owner: str, repo: str, max_depth: int = 2) -> list[str]:
        """
        Fetch repository file tree (top-level files and directories).
        
        Args:
            owner: Repository owner
            repo: Repository name
            max_depth: Maximum depth to traverse (default: 2)
            
        Returns:
            List of file paths
        """
        cache_key = f"github:{owner}/{repo}:tree"
        cached = github_cache.get(cache_key)
        if cached is not None:
            return cached

        client = await self._get_client()
        
        # Get the default branch first
        repo_info = await self.get_repo_info(owner, repo)
        default_branch = repo_info.get("default_branch", "main")

        # Fetch the tree recursively
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"

        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            # Extract file paths, limiting depth
            files = []
            for item in data.get("tree", []):
                path = item.get("path", "")
                # Count depth by number of slashes
                depth = path.count("/")
                if depth < max_depth:
                    # Mark directories with trailing slash
                    if item.get("type") == "tree":
                        path += "/"
                    files.append(path)

            # Sort for consistent output
            files.sort()
            
            github_cache.set(cache_key, files, settings.cache_ttl_seconds)
            return files

        except httpx.HTTPStatusError:
            # Fallback to root contents if tree API fails
            url = f"{self.BASE_URL}/repos/{owner}/{repo}/contents"
            response = await client.get(url)
            response.raise_for_status()
            
            files = [
                item["name"] + ("/" if item["type"] == "dir" else "")
                for item in response.json()
            ]
            files.sort()
            
            github_cache.set(cache_key, files, settings.cache_ttl_seconds)
            return files

    async def get_readme(self, owner: str, repo: str) -> str:
        """
        Fetch and decode repository README.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            README content (truncated to max_chars)
        """
        cache_key = f"github:{owner}/{repo}:readme"
        cached = github_cache.get(cache_key)
        if cached is not None:
            return cached

        client = await self._get_client()
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/readme"

        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            # Decode base64 content
            content = data.get("content", "")
            if content:
                readme = base64.b64decode(content).decode("utf-8", errors="ignore")
                # Clean up markdown for better analysis
                readme = self._clean_markdown(readme)
                # Truncate to configured max length
                readme = readme[: settings.readme_max_chars]
            else:
                readme = ""

            github_cache.set(cache_key, readme, settings.cache_ttl_seconds)
            return readme

        except httpx.HTTPStatusError:
            # No README found
            github_cache.set(cache_key, "", settings.cache_ttl_seconds)
            return ""

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """
        Clean markdown text for better LLM analysis.
        
        Removes:
        - Code blocks
        - Inline code
        - Links (keeps text)
        - Headers (keeps text)
        - Excessive whitespace
        """
        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)
        # Remove inline code
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # Remove links, keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove header markers
        text = re.sub(r"#{1,6}\s+", "", text)
        # Remove image references
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
        # Normalize whitespace
        text = " ".join(text.split())

        return text

    async def investigate_repo(self, owner: str, repo: str) -> RepoEvidence:
        """
        Perform full repository investigation with parallel fetches.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            RepoEvidence with file tree, README, and description
        """
        # Fetch all data in parallel
        info, tree, readme = await asyncio.gather(
            self.get_repo_info(owner, repo),
            self.get_file_tree(owner, repo),
            self.get_readme(owner, repo),
        )

        return RepoEvidence(
            owner=owner,
            repo=repo,
            file_tree=tree,
            readme=readme,
            description=info.get("description"),
        )


# Global client instance
github_client = GitHubClient()

