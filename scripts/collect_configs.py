# scripts/collect_configs.py
from pathlib import Path
import json
import yaml

agg = {}
for p in Path("configs").rglob("*.y*ml"):
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    agg[str(p)] = data

build_dir = Path("build")
build_dir.mkdir(parents=True, exist_ok=True)
Path(build_dir / "config_snapshot.json").write_text(json.dumps(agg, indent=2))