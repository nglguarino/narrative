"""
Article-level narrative extraction.

Extracts narratives directly from full articles using multi-agent consensus.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from ..models.agents import Agent, MultiAgentConsensus
from ..models.async_agents import AsyncMultiAgentConsensus
from ..utils.deduplication import SemanticDedupe


class ArticleNarrativeExtractor:
    """Extract narratives directly from full articles using multi-agent consensus."""

    # CHANGE: Updated to target the abstraction level of PolyNarrative's sub-narratives
    # (Recurring argumentative patterns rather than hyper-specific instances)
    # CHANGE: Shifted from "Abstract Categories" to "Propositional Claims"
    # to match the granularity of your Fine-Grained Gold Labels.
    SYSTEM_PROMPT = """You are an expert propaganda analyst identifying specific underlying narratives in news articles.

        Definition: A narrative is a specific underlying premise, claim, or argument being advanced by the text. 

        CRITICAL INSTRUCTION: Do NOT generate broad topics (e.g., "Criticism of climate policies"). Instead, extract the SPECIFIC argument being made (e.g., "Climate policies hurt the economy").

        Guidelines:
        1. Formulate narratives as DECLARATIVE STATEMENTS (Sentences), not Noun Phrases.
        2. Capture the specific "Cause & Effect" or "Judgment" being proposed.
        3. If the text criticizes an entity, specify WHAT the criticism claims (e.g., instead of "Criticism of Ukraine", output "Ukraine is a hub for criminal activities").
        4. Be concise but specific (5-15 words).

        Examples of CORRECT Narrative Extraction (Fine-Grained):
        - "Sanctions imposed by Western countries will backfire" (NOT "Sanctions analysis")
        - "The West does not care about Ukraine, only about its interests" (NOT "Western geopolitics")
        - "Renewable energy is unreliable" (NOT "Green energy criticism")
        - "Climate agenda has hidden motives" (NOT "Conspiracy theories")
        - "NATO will destroy Russia" (NOT "Anti-NATO sentiment")

        Output Format:
        Output ONLY the narrative statements, one per line, no numbering or bullets."""

    def __init__(self, agents: List[Agent], embedding_model=None, max_narratives: int = None):
        """
        Initialize article extractor.

        Args:
            agents: List of LLM agents
            embedding_model: Optional embedding model for semantic deduplication
        """
        self.agents = agents
        self.consensus = AsyncMultiAgentConsensus(agents)
        self.embedding_model = embedding_model
        self.max_narratives = max_narratives
        if embedding_model:
            self.semantic_dedupe = SemanticDedupe(
                embedding_model,
                similarity_threshold=0.85  # Higher threshold for brief narratives
            )
        else:
            self.semantic_dedupe = None

    def extract_from_article(self, article: str, article_id: Any = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract narratives from a full article.

        Args:
            article: Full article text
            article_id: Optional identifier for the article
            metadata: Optional metadata (date, source, actors, etc.)

        Returns:
            Dictionary with article narratives and metadata
        """
        print(f"Processing article {article_id}")

        # Create extraction prompt
        prompt = self._create_extraction_prompt(article, metadata)

        # Get narratives from all agents
        raw_narratives = self.consensus.generate_with_consensus(
            prompt,
            self.SYSTEM_PROMPT
        )

        # Deduplicate semantically if embedding model available
        if self.semantic_dedupe:
            narratives = self.semantic_dedupe.deduplicate(raw_narratives)
        else:
            narratives = list(set(raw_narratives))

        # Apply max narratives limit if configured
        if self.max_narratives:
            narratives = narratives[:self.max_narratives]

        # Filter for brevity (relaxed slightly to allow for fine-grained detail)
        narratives = [n for n in narratives if len(n.split()) >= 2 and len(n.split()) <= 20]

        return {
            'article_id': article_id,
            'narratives': narratives,
            'raw_narrative_count': len(raw_narratives),
            'article_text': article[:500] + "..." if len(article) > 500 else article,
            'metadata': metadata or {}
        }

    def _create_extraction_prompt(self, article: str, metadata: Optional[Dict[str, Any]]) -> str:
        context = ""
        if metadata:
            if 'date' in metadata:
                context += f"\nDate: {metadata['date']}"
            if 'source' in metadata:
                context += f"\nSource: {metadata['source']}"
            if 'title' in metadata:
                context += f"\nTitle: {metadata['title']}"

        prompt = f"""Identify the narratives used in this article.{context}

    ARTICLE:
    {article}
    """

        return prompt

    def batch_extract(self, articles: List[str], article_ids: List[Any] = None,
                     metadata_list: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract narratives from multiple articles.

        Args:
            articles: List of article texts
            article_ids: Optional list of article identifiers
            metadata_list: Optional list of metadata dictionaries

        Returns:
            List of article results
        """
        if article_ids is None:
            article_ids = list(range(len(articles)))

        if metadata_list is None:
            metadata_list = [{}] * len(articles)

        results = []
        for article, article_id, metadata in zip(articles, article_ids, metadata_list):
            result = self.extract_from_article(article, article_id, metadata)
            results.append(result)

        return results