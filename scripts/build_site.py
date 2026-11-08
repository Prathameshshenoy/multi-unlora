"""Build the dependency-free static visualizer for optional Sites hosting."""

from pathlib import Path
import shutil

root = Path(__file__).resolve().parent.parent
web = root / "web"
output = web / "dist"
output.mkdir(exist_ok=True)
for name in ("index.html", "styles.css", "app.js"):
    shutil.copy2(web / name, output / name)
print(f"Built static visualizer: {output}")
