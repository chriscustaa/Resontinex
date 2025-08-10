#!/usr/bin/env python3
from tune_overlay import ParameterOptimizer, GridSearchTuner, ScenarioManager
import yaml, json
from pathlib import Path

def main():
    params = yaml.safe_load(open("configs/fusion/overlay_params.yaml"))
    scenarios = yaml.safe_load(open("configs/fusion/eval_scenarios.yaml"))
    sm = ScenarioManager(scenarios)
    tuner = GridSearchTuner()
    opt = ParameterOptimizer(params)
    for sid in sm.ids():
        best = opt.optimize(sid, tuner.grid("small"))
        params = opt.persist(params, sid, best)
    Path("build/tuning").mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(params, open("configs/fusion/overlay_params.yaml","w"))
    json.dump({"ok":True}, open("build/tuning/summary.json","w"))

if __name__ == "__main__":
    main()