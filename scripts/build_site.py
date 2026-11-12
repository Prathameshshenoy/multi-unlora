"""Build the static visualizer and its source-linked benchmark summaries."""

from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from scripts.sync_benchmarks import sync

sync(root)
web = root / "web"
output = web / "dist"
if output.exists():
    shutil.rmtree(output)
output.mkdir()
for name in ("index.html", "styles.css", "app.js"):
    shutil.copy2(web / name, output / name)
(output / "results").mkdir()
shutil.copy2(web / "results" / "benchmarks.json", output / "results" / "benchmarks.json")
print(f"Built static visualizer: {output}")
