"""Re-export InvertedIndex so lib.hybrid_search can use a relative import."""
import sys
from pathlib import Path

# Ensure cli/ is importable when loaded as a package module
_cli_dir = str(Path(__file__).resolve().parent.parent)
if _cli_dir not in sys.path:
    sys.path.insert(0, _cli_dir)

from inverted_index import InvertedIndex  # noqa: E402

__all__ = ["InvertedIndex"]
