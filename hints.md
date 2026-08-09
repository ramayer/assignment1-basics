# Hints

## On windows python

You need to `export PYTHONUTF8=1` for the Byte Pair Encoder training to work.

## For testing

Use `uv run pytest -k 'test_train_bpe'` to run just tests that start with train_bpe.

To lint and reformat python without messing up theirs, `hatch fmt cs336_basics/ron_*.py`
or `uv run ruff check cs336_basics/ron_*.py`


