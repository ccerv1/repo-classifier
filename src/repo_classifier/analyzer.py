"""
LLM-powered repository analyzer using OpenAI structured outputs.
Synthesizes classification from evidence and historical precedents.
"""

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from repo_classifier.config import settings
from repo_classifier.github_client import RepoEvidence
from repo_classifier.rag import Precedent


class AnalysisResult(BaseModel):
    """Structured output from the LLM analysis."""
    
    category: str = Field(
        description="The category that best matches this repository"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score from 0 to 1"
    )
    reasoning: str = Field(
        description="Brief explanation of why this category was chosen"
    )
    similar_precedents: list[str] = Field(
        default_factory=list,
        description="URLs of similar repositories that influenced the decision"
    )


class Analyzer:
    """
    Repository analyzer using OpenAI for classification.
    
    Uses structured outputs (response_format with json_schema) to guarantee
    valid JSON responses matching the AnalysisResult schema.
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def analyze(
        self,
        evidence: RepoEvidence,
        precedents: list[Precedent],
        categories: dict[str, str],
    ) -> AnalysisResult:
        """
        Analyze a repository and classify it.
        
        Args:
            evidence: Evidence collected from the repository
            precedents: Similar repositories from RAG store
            categories: Available categories with descriptions
            
        Returns:
            AnalysisResult with category, confidence, and reasoning
        """
        # Build the system prompt
        system_prompt = self._build_system_prompt(categories)
        
        # Build the user prompt with evidence and precedents
        user_prompt = self._build_user_prompt(evidence, precedents, categories)
        
        # Call OpenAI with structured output
        response = await self.client.chat.completions.create(
            model=settings.llm_model,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "analysis_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "The category that best matches this repository"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence score from 0 to 1"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of why this category was chosen"
                            },
                            "similar_precedents": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "URLs of similar repositories that influenced the decision"
                            }
                        },
                        "required": ["category", "confidence", "reasoning", "similar_precedents"],
                        "additionalProperties": False
                    }
                }
            },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent classification
        )
        
        # Parse response
        content = response.choices[0].message.content
        result = AnalysisResult.model_validate_json(content)
        
        # Validate category is one of the provided options
        if result.category not in categories:
            # Find best match or use first category
            result.category = self._find_closest_category(result.category, categories)
        
        return result

    def _build_system_prompt(self, categories: dict[str, str]) -> str:
        """Build the system prompt for classification."""
        category_list = "\n".join(
            f"- **{name}**: {description}"
            for name, description in categories.items()
        )
        
        return f"""You are a repository classification expert. Your job is to analyze GitHub repositories and classify them into one of the provided categories.

## Available Categories

{category_list}

## Instructions

1. Analyze the evidence provided (file tree, README, description)
2. Consider the similar repositories (precedents) and their categories
3. Choose the SINGLE best-matching category
4. Provide a confidence score (0.0 to 1.0) based on how well the evidence supports your choice
5. Explain your reasoning concisely

## Rules

- You MUST choose exactly one category from the provided list
- Be specific in your reasoning, citing evidence from the file tree or README
- If precedents are similar, weight their categories in your decision
- Confidence should reflect certainty: >0.9 = very confident, 0.7-0.9 = confident, <0.7 = uncertain"""

    def _build_user_prompt(
        self,
        evidence: RepoEvidence,
        precedents: list[Precedent],
        categories: dict[str, str],
    ) -> str:
        """Build the user prompt with evidence and precedents."""
        sections = []
        
        # Repository info
        sections.append(f"## Repository: {evidence.owner}/{evidence.repo}")
        
        # Description
        if evidence.description:
            sections.append(f"### Description\n{evidence.description}")
        
        # File tree (limit to relevant files)
        if evidence.file_tree:
            relevant_files = self._filter_relevant_files(evidence.file_tree)
            if relevant_files:
                file_list = "\n".join(f"- {f}" for f in relevant_files[:50])
                sections.append(f"### File Tree (key files)\n{file_list}")
        
        # README (truncated)
        if evidence.readme:
            readme_preview = evidence.readme[:3000]
            if len(evidence.readme) > 3000:
                readme_preview += "\n...[truncated]"
            sections.append(f"### README\n{readme_preview}")
        
        # Precedents (similar repositories)
        if precedents:
            precedent_list = "\n".join(
                f"- {p.url} → **{p.category}** (similarity: {p.similarity:.2f})"
                for p in precedents
            )
            sections.append(f"### Similar Repositories (from knowledge base)\n{precedent_list}")
        else:
            sections.append("### Similar Repositories\nNo similar repositories found in knowledge base.")
        
        # Categories reminder
        category_names = ", ".join(categories.keys())
        sections.append(f"### Available Categories\n{category_names}")
        
        return "\n\n".join(sections)

    def _filter_relevant_files(self, files: list[str]) -> list[str]:
        """Filter file tree to show only relevant/interesting files."""
        # Files that indicate project type
        interesting_patterns = [
            # Package manifests
            "package.json", "pyproject.toml", "Cargo.toml", "go.mod",
            "pom.xml", "build.gradle", "Gemfile", "composer.json",
            # Config files
            "Dockerfile", "docker-compose", ".github/", "Makefile",
            "tsconfig.json", "webpack", "vite.config", "next.config",
            # Source directories
            "src/", "lib/", "app/", "cmd/", "pkg/", "internal/",
            # Framework indicators
            "pages/", "components/", "routes/", "models/", "views/",
            # ML/Data
            "notebooks/", "data/", "train", "model", ".ipynb",
            # Mobile
            "ios/", "android/", "App.tsx", "App.js",
            # Docs
            "README", "docs/", "LICENSE",
        ]
        
        relevant = []
        for f in files:
            f_lower = f.lower()
            if any(pattern.lower() in f_lower for pattern in interesting_patterns):
                relevant.append(f)
            elif f.count("/") == 0:  # Root-level files
                relevant.append(f)
        
        return relevant

    def _find_closest_category(
        self,
        category: str,
        valid_categories: dict[str, str],
    ) -> str:
        """Find the closest matching category if exact match not found."""
        category_lower = category.lower()
        
        for name in valid_categories:
            if category_lower in name.lower() or name.lower() in category_lower:
                return name
        
        # Return first category as fallback
        return next(iter(valid_categories))


# Global analyzer instance
analyzer = Analyzer()

