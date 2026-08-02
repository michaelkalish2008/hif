"""Per-signal visualization generators. One module per HIF signal.

Each module exports:
- ``LABEL``, ``GLYPH``
- ``available(profile) -> str | None`` — None if the signal's backing data is
  present in this profile, else a human-readable reason it's unavailable.
- ``generate(profile, output_path, formats) -> dict[str, Path]``
"""
