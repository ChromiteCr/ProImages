import pytest

from proimages.api.lifespan import max_concurrent_jobs


@pytest.mark.parametrize(("value", "expected"), [(None, 1), ("1", 1), ("3", 3)])
def test_max_concurrent_jobs_reads_proimages_max_jobs(value, expected, monkeypatch):
    if value is None:
        monkeypatch.delenv("PROIMAGES_MAX_JOBS", raising=False)
    else:
        monkeypatch.setenv("PROIMAGES_MAX_JOBS", value)

    assert max_concurrent_jobs() == expected


@pytest.mark.parametrize("value", ["0", "-2", "two", ""])
def test_max_concurrent_jobs_rejects_anything_but_a_positive_integer(value, monkeypatch):
    monkeypatch.setenv("PROIMAGES_MAX_JOBS", value)

    with pytest.raises(ValueError, match="PROIMAGES_MAX_JOBS must be a positive integer"):
        max_concurrent_jobs()
