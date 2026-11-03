"""Read-only figure freshness checks, usable without importing a CLI.

The original renderer files remain unchanged because existing figures bind
their exact bytes. Keep the original filename/content fingerprint convention.
"""

import hashlib
from pathlib import Path

from .common import digest

GENERATOR_DIRECTORY = Path(__file__).with_name("test02d")


def generator_sha(directory=None):
    directory = Path(directory) if directory is not None else GENERATOR_DIRECTORY
    checksum = hashlib.sha256()
    for path in sorted(directory.glob("*.py")):
        checksum.update(path.name.encode() + b"\0" + path.read_bytes())
    return checksum.hexdigest()


def stale(manifest, *, generator_directory=None):
    """Return every generator, source or output mismatch in manifest order."""
    reasons = []
    current = generator_sha(generator_directory)
    for figure in manifest["figures"]:
        if figure["generator_sha256"] != current:
            reasons.append(figure["figure_id"] + ": generator changed")
        for key, label in (("sources", "source"), ("outputs", "output")):
            for item in figure[key]:
                path = Path(item["path"])
                if not path.is_file() or digest(path) != item["sha256"]:
                    reasons.append(figure["figure_id"] + f": {label} changed " + str(path))
    return reasons
