"""
Async multi-agent consensus for parallel API calls.
Makes agent calls happen simultaneously instead of sequentially.
"""

import asyncio
from typing import List


class AsyncMultiAgentConsensus:
    """Manages consensus-based generation with parallel async calls."""

    def __init__(self, agents):
        self.agents = agents

    async def generate_async(self, agent, prompt: str, system_prompt: str):
        """Generate response from single agent asynchronously."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: agent.generate(prompt, system_prompt)
            )
            return self._parse_narratives(response)
        except Exception as e:
            print(f"Warning: Agent {agent.name} failed: {e}")
            return []

    async def generate_with_consensus_async(self, prompt: str, system_prompt: str):
        """Generate responses from all agents in parallel."""
        # Create tasks for all agents
        tasks = [
            self.generate_async(agent, prompt, system_prompt)
            for agent in self.agents
        ]

        # Run all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all narratives
        all_narratives = []
        for result in results:
            if isinstance(result, list):
                all_narratives.extend(result)
            elif isinstance(result, Exception):
                print(f"Warning: Task failed with {result}")

        return all_narratives

    def generate_with_consensus(self, prompt: str, system_prompt: str):
        """Synchronous wrapper for async generation."""
        # Import nest_asyncio to handle nested event loops
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass

        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run async function
        return loop.run_until_complete(
            self.generate_with_consensus_async(prompt, system_prompt)
        )

    def _parse_narratives(self, response: str):
        """Parse narrative strings from agent response."""
        narratives = []
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove numbering (1., 2., etc.)
            if line and line[0].isdigit():
                idx = 0
                while idx < len(line) and (line[idx].isdigit() or line[idx] in ['.', ')', ']']):
                    idx += 1
                line = line[idx:].strip()

            # Remove bullet points
            if line.startswith('- ') or line.startswith('• ') or line.startswith('* '):
                line = line[2:].strip()

            # Remove quotes if present
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1].strip()

            if line and len(line) > 10:
                narratives.append(line)

        return narratives