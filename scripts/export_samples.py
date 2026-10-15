"""Write small, deterministic samples of every twin dataset to sample_data/.

These are the same generators the platform trains on, so the samples show
exactly what the models see. Regenerate with: python scripts/export_samples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from railsentinel.twin import generators  # noqa: E402
from railsentinel.twin.network import load_network  # noqa: E402
from railsentinel.twin.world import WorldParams  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "sample_data"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    net = load_network()
    w = WorldParams()
    generators.confirmation_samples(net, 300, np.random.default_rng(1), w).to_csv(OUT / "confirmation_outcomes.csv",
                                                                                  index=False)
    generators.delay_samples(net, 300, np.random.default_rng(2), w).to_csv(OUT / "train_delays.csv", index=False)
    generators.fare_samples(net, 300, np.random.default_rng(3), w).to_csv(OUT / "fare_occupancy.csv", index=False)
    generators.fraud_sessions(500, np.random.default_rng(4), w).to_csv(OUT / "booking_sessions.csv", index=False)
    episodes = generators.recommendation_episodes(net, 20, np.random.default_rng(5), w)
    with (OUT / "recommendation_choices.jsonl").open("w", encoding="utf-8") as fh:
        for ep in episodes:
            fh.write(json.dumps({k: v for k, v in ep.items() if k != "latent_w"}) + "\n")
    trains = [t.to_dict() for t in list(net.trains.values())]
    (OUT / "timetable.json").write_text(json.dumps(trains, indent=1), encoding="utf-8")
    print(f"Wrote samples to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
