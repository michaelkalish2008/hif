"""Application archetype registry.

Each archetype is a flat YAML file in this directory with a description and a
default analysis window. Selecting one via ``--application`` labels the run
(the archetype id and the effective analysis window are recorded in the JSON
record's extras) and fills in ``--analysis-window`` when the user did not pass
one. It does not change how the prompt is perturbed or how any measurement is
computed. (Earlier revisions declared a per-archetype ``perturbation_family``
and ``report_template``; nothing ever consumed them, so the registry no
longer carries fields the pipeline does not read.)

The YAML files use a flat ``key: value`` format, parsed with a minimal
hand-rolled parser so the core package does not depend on pyyaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

_ARCHETYPES_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Archetype:
    id: str
    description: str
    default_analysis_window: Union[int, str]  # int token count or "adaptive"


class UnknownArchetypeError(KeyError):
    """Raised when an archetype id has no registry entry."""

    def __init__(self, archetype_id: str, valid_ids: list[str]):
        self.archetype_id = archetype_id
        self.valid_ids = valid_ids
        super().__init__(
            f"Unknown archetype {archetype_id!r}. Valid: {', '.join(valid_ids)}"
        )


def _parse_flat_yaml(text: str) -> dict[str, object]:
    """Parse a flat ``key: value`` YAML subset (no nesting, no lists)."""
    data: dict[str, object] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Malformed archetype line (expected 'key: value'): {raw_line!r}")
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        # Strip inline comments outside quotes
        if value and value[0] not in "\"'" and " #" in value:
            value = value.split(" #", 1)[0].strip()
        # Quoted strings
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            data[key] = value[1:-1]
            continue
        if value in ("null", "~", ""):
            data[key] = None
        elif value.lstrip("-").isdigit():
            data[key] = int(value)
        elif value in ("true", "false"):
            data[key] = value == "true"
        else:
            data[key] = value
    return data


def list_archetypes() -> list[str]:
    """Return sorted archetype ids present in the registry."""
    return sorted(p.stem for p in _ARCHETYPES_DIR.glob("*.yaml"))


def load_archetype(archetype_id: str) -> Archetype:
    """Load an archetype definition by id.

    Raises UnknownArchetypeError if no yaml file exists for the id.
    """
    path = _ARCHETYPES_DIR / f"{archetype_id}.yaml"
    if not path.exists():
        raise UnknownArchetypeError(archetype_id, list_archetypes())
    data = _parse_flat_yaml(path.read_text(encoding="utf-8"))
    required = ("id", "description", "default_analysis_window")
    missing = [k for k in required if data.get(k) is None]
    if missing:
        raise ValueError(f"Archetype file {path.name} missing fields: {', '.join(missing)}")
    return Archetype(
        id=str(data["id"]),
        description=str(data["description"]),
        default_analysis_window=data["default_analysis_window"],  # type: ignore[arg-type]
    )
