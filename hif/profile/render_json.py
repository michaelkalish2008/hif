"""JSON serialization and file-writing for BehavioralRangeProfile artifacts."""

from pathlib import Path

from hif.profile.schema import BehavioralRangeProfile


def render_json(profile: BehavioralRangeProfile, output_path: Path) -> None:
    """Write the profile to disk as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(profile.model_dump_json(indent=2))
