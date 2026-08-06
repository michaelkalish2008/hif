"""`python -m hif.cli` — the same entry point as the `hif` script.

When the CLI was one module, `if __name__ == "__main__": app()` at its foot
did this. A package needs the guard in its own file, and the tests and docs
that invoke `python -m hif.cli` depend on it.
"""

from hif.cli import app

if __name__ == "__main__":
    app()
