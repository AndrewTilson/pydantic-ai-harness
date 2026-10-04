"""Minimum-token suffix selection shared by summarizing and sliding-window compaction."""

from __future__ import annotations

from collections.abc import Callable

from pydantic_ai.messages import ModelMessage

from pydantic_ai_harness.compaction._shared import estimate_token_count, find_safe_cutoff


def validate_min_keep_tokens(min_keep_tokens: int | None, keep_tokens: int | None) -> None:
    if min_keep_tokens is None:
        return
    if min_keep_tokens <= 0:
        raise ValueError('min_keep_tokens must be positive.')
    if keep_tokens is not None:
        raise ValueError('min_keep_tokens and keep_tokens are mutually exclusive.')


def find_minimum_token_cutoff(
    messages: list[ModelMessage],
    min_tokens: int,
    tokenizer: Callable[[str], int] | None,
) -> int:
    """Keep the shortest whole-message suffix reaching `min_tokens`, then protect tool pairs.

    Counts use the same estimator as compaction triggers. If the entire history is below
    the minimum, there is no prefix to compact. Whole messages and tool pairs can make the
    retained suffix exceed the minimum; this is not a model context-window limit.
    """
    if estimate_token_count(messages, tokenizer) <= min_tokens:
        return 0

    lo, hi = 0, len(messages)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if estimate_token_count(messages[mid:], tokenizer) >= min_tokens:
            lo = mid
        else:
            hi = mid

    # A long-running tool may return more than the default search range after its call.
    return find_safe_cutoff(messages, len(messages) - lo, search_range=len(messages))
