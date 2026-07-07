from harness.discord_adapter import _drain_log_batch


def test_drain_log_batch_joins_messages_with_separator() -> None:
    chunks, remaining = _drain_log_batch(
        ["alpha", "beta", "gamma"],
        max_length=50,
        max_chunks=2,
    )

    assert chunks == ["alpha\n---\nbeta\n---\ngamma"]
    assert remaining == []


def test_drain_log_batch_sends_at_most_two_chunks_and_carries_rest() -> None:
    chunks, remaining = _drain_log_batch(
        ["alpha", "beta", "gamma", "delta", "epsilon"],
        max_length=20,
        max_chunks=2,
    )

    assert chunks == ["alpha\n---\nbeta", "gamma\n---\ndelta"]
    assert remaining == ["epsilon"]


def test_drain_log_batch_splits_oversized_message_and_carries_tail() -> None:
    chunks, remaining = _drain_log_batch(
        ["abcdefghijklmno", "tail"],
        max_length=10,
        max_chunks=2,
    )

    assert chunks == ["abcdefghij", "klmno"]
    assert remaining == ["tail"]


def test_drain_log_batch_keeps_all_chunks_within_limit() -> None:
    chunks, remaining = _drain_log_batch(
        ["a" * 8, "b" * 8, "c" * 8],
        max_length=10,
        max_chunks=2,
    )

    assert chunks == ["a" * 8, "b" * 8]
    assert remaining == ["c" * 8]
