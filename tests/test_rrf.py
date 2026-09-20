"""RRF fusion and tokenization unit tests."""

import pytest

from rag.retrieval.hybrid import rrf_fuse, tokenize


def test_rrf_score_math():
    fused = rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["d"] == pytest.approx(1 / 63)


def test_rrf_prefers_chunks_ranked_well_by_both_retrievers():
    fused = rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60)
    ranking = sorted(fused, key=fused.get, reverse=True)
    # b and c appear in both rankings; a and d in only one.
    assert ranking[:2] == ["b", "c"]
    assert set(ranking[2:]) == {"a", "d"}


def test_rrf_with_single_ranking_matches_rank_order():
    fused = rrf_fuse([["x", "y"]], k=60)
    assert fused["x"] > fused["y"]


def test_tokenize_lowercases_and_drops_single_characters():
    tokens = tokenize("The FWBI1 index (R2 = 0.639, p < 0.05)")
    # Single characters (r, p, 0) are formula/page noise and get dropped.
    assert tokens == ["the", "fwbi1", "index", "639", "05"]


def test_tokenize_empty_query():
    assert tokenize("   ") == []
