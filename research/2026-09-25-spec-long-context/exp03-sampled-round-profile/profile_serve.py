"""`lmk serve` with profile_patch installed first (same process, so the patch reaches the engine)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_patch  # noqa: E402

from lmk.cli import main  # noqa: E402

profile_patch.install()
sys.exit(main(["serve"]))
