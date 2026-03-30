"""Built-in pricing table for common LLM models.

Costs are in USD per 1 million tokens: (input_cost, output_cost).
"""

from __future__ import annotations

# (input_cost_per_1m_tokens, output_cost_per_1m_tokens)
MODEL_COSTS: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    # Anthropic
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-haiku-3.5": (0.80, 4.00),
    "claude-haiku-3-5-20241022": (0.80, 4.00),
}

# Aliases: map short names to canonical names above
_ALIASES: dict[str, str] = {
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-opus-4": "claude-opus-4-20250514",
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float | None:
    """Return estimated cost in USD, or None if model is unknown."""
    resolved = _ALIASES.get(model, model)
    costs = MODEL_COSTS.get(resolved)
    if costs is None:
        # Try prefix matching for versioned model strings
        for key, val in MODEL_COSTS.items():
            if resolved.startswith(key):
                costs = val
                break
    if costs is None:
        return None
    input_cost, output_cost = costs
    return (tokens_in * input_cost + tokens_out * output_cost) / 1_000_000
