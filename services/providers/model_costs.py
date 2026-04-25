"""
Centralized per-token pricing for all supported models.

All costs are in USD per million tokens.
Input  = cost to process the prompt/context
Output = cost to generate the response

Usage:
    from model_costs import COSTS, estimate_cost
    cost = estimate_cost("anthropic/claude-haiku-4.5-20241022", input_tokens=1000, output_tokens=500)
    # => 0.0015  ($0.001 input + $0.0025 output)
"""

# ── Pricing: $ per 1,000,000 tokens ──────────────────────────────

COSTS: dict[str, tuple[float, float]] = {
    # ── Anthropic ────────────────────────────────────────────────
    # Model                                         (input, output)
    "claude-haiku-4.5":                (1.00,  5.00),  # Claude 4.5 Haiku / 3.5 Haiku
    "claude-haiku-4-5":                (1.00,  5.00),
    "claude-haiku-3.5":                (1.00,  5.00),
    "claude-sonnet-4":                 (3.00, 15.00),  # Claude 4 Sonnet
    "claude-sonnet-4-20250514":        (3.00, 15.00),
    "claude-sonnet-4-20250514:20250514": (3.00, 15.00),
    "claude-sonnet-4-20250514-thinking": (3.00, 15.00),
    "claude-sonnet-4-6":               (3.00, 15.00),
    "claude-sonnet-4-20250514-6":      (3.00, 15.00),
    "claude-sonnet-4.5":               (3.00, 15.00),
    "claude-sonnet-4-5":               (3.00, 15.00),
    "claude-sonnet-4-20250514-5":      (3.00, 15.00),
    "claude-sonnet-4-20250514-6":      (3.00, 15.00),
    "claude-opus-4":                   (15.00, 75.00),  # Claude 4 Opus
    "claude-opus-4-20250514":          (15.00, 75.00),
    "claude-opus-4-6":                 (15.00, 75.00),
    "claude-opus-4-20250514-6":        (15.00, 75.00),
    "sonnet":                          (3.00, 15.00),   # Claude short names
    "opus":                            (15.00, 75.00),
    "haiku":                           (1.00,  5.00),

    # ── OpenAI ──────────────────────────────────────────────────
    # Model                                         (input, output)
    "gpt-5.4":                         (1.25, 10.50),  # GPT-5.4 (latest flagship)
    "gpt-5.4-mini":                    (0.75,  4.50),  # GPT-5.4 Mini
    "gpt-5-mini":                      (0.25,  2.00),  # GPT-5 Mini (budget)
    "gpt-4.1":                         (2.00,  8.00),  # GPT-4.1
    "gpt-4.1-mini":                    (0.40,  1.60),  # GPT-4.1 Mini
    "gpt-4.1-nano":                    (0.10,  0.40),  # GPT-4.1 Nano (cheapest GPT)
    "gpt-4o":                          (2.50, 10.00),  # GPT-4o
    "gpt-4o-mini":                     (0.15,  0.60),  # GPT-4o Mini
    "o3":                              (10.50, 42.00),  # O3 (reasoning)
    "o3-mini":                         (1.10,  4.40),  # O3 Mini
    "o4-mini":                         (1.10,  4.40),  # O4 Mini

    # ── Google Gemini ───────────────────────────────────────────
    # Model                                         (input, output)
    "gemini-3.1-pro-preview":          (1.25, 10.00),  # Gemini 3.1 Pro
    "gemini-2.5-pro":                  (1.25, 10.00),  # Gemini 2.5 Pro
    "gemini-2.5-flash":                (0.30,  2.50),  # Gemini 2.5 Flash
    "gemini-2.5-flash-lite":           (0.08,  0.60),  # Gemini 2.5 Flash Lite (free tier eligible)
    "gemini-2.0-flash":                (0.10,  1.00),  # Gemini 2.0 Flash
    "gemini-2.0-flash-lite":           (0.08,  0.60),  # Gemini 2.0 Flash Lite
    "gemini-1.5-pro":                  (1.25,  5.00),  # Gemini 1.5 Pro
    "gemini-1.5-flash":                (0.08,  0.30),  # Gemini 1.5 Flash

    # ── OpenCode model aliases (full paths) ─────────────────────
    "anthropic/claude-sonnet-4-20250514":     (3.00, 15.00),
    "anthropic/claude-opus-4-20250514":       (15.00, 75.00),
    "anthropic/claude-haiku-4.5-20241022":    (1.00,  5.00),
    "openai/gpt-4.1":                        (2.00,  8.00),
    "openai/gpt-4.1-mini":                   (0.40,  1.60),
    "openai/o3":                             (10.50, 42.00),
    "openai/o4-mini":                        (1.10,  4.40),
    "google/gemini-2.5-flash":               (0.30,  2.50),
    "google/gemini-2.5-pro":                 (1.25, 10.00),
}


def estimate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> float:
    """
    Estimate cost in USD for a given model and token usage.

    Cache pricing (Anthropic):
      - cache_write (creation): 1.25x normal input rate
      - cache_read: 0.1x normal input rate

    Args:
        model: Model identifier string (any known variant)
        input_tokens: Prompt/context tokens consumed
        output_tokens: Generated/completion tokens produced
        cache_read_tokens: Tokens served from cache (discounted)
        cache_write_tokens: Tokens written to cache (premium)
        reasoning_tokens: Tokens spent on internal reasoning (GPT-o, Gemini)

    Returns:
        Estimated cost in USD (0.0 for unknown/local models)
    """
    model_lower = model.lower().strip()

    # Try exact match first
    rates = COSTS.get(model_lower)

    # Fuzzy match: try to find a key that's contained in the model name
    if rates is None:
        for key, val in COSTS.items():
            if key in model_lower or model_lower in key:
                rates = val
                break

    if rates is None:
        # Unknown / local model — cost is $0
        return 0.0

    input_rate, output_rate = rates  # per 1M tokens

    cost = 0.0
    cost += (input_tokens / 1_000_000) * input_rate
    cost += (output_tokens / 1_000_000) * output_rate

    # Cache adjustments (Anthropic-style)
    if cache_read_tokens:
        cost += (cache_read_tokens / 1_000_000) * (input_rate * 0.1)
    if cache_write_tokens:
        cost += (cache_write_tokens / 1_000_000) * (input_rate * 1.25)

    # Reasoning tokens billed as output tokens (OpenAI o-series pattern)
    if reasoning_tokens:
        cost += (reasoning_tokens / 1_000_000) * output_rate

    return round(cost, 8)


def get_model_rates(model: str) -> tuple[float, float] | None:
    """
    Look up per-million-token rates for a model.
    Returns (input_rate, output_rate) or None if unknown.
    """
    model_lower = model.lower().strip()
    rates = COSTS.get(model_lower)
    if rates is None:
        for key, val in COSTS.items():
            if key in model_lower or model_lower in key:
                return val
    return rates
