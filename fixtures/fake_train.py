"""Stand-in solution for exercising the harness without real data or a real
model. Honors the same invocation contract as a real solution:

    python fake_train.py --config <cfg> --seed <n> --out <result.json>

`<cfg>` is read as JSON (not YAML -- this is a fixture, not a real solution,
and stdlib json keeps it dependency-free) with these optional keys:

    mode:    "normal" (default) | "timeout" | "crash" | "bad_output"
    mean:    float, mean of the simulated validation primary (default 0.59)
    std:     float, stdev of the simulated validation primary (default 0.01)
    sleep_s: float, simulated training time (default 2.0)
    epochs:  int, reported epochs_run (default 10)

Runs standalone in a subprocess with no access to the repo's packages, so the
TEST_METRICS sentinel below is a literal string, not an import from
agent.config.TEST_METRICS_SENTINEL -- keep the two in sync by hand.
"""
import argparse
import json
import random
import sys
import time

TEST_METRICS_SENTINEL = "TEST_METRICS:"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = json.loads(open(a.config).read()) if a.config else {}
    mode = cfg.get("mode", "normal")
    mean = cfg.get("mean", 0.59)
    std = cfg.get("std", 0.01)
    sleep_s = cfg.get("sleep_s", 2.0)
    epochs = cfg.get("epochs", 10)

    if mode == "crash":
        time.sleep(min(sleep_s, 0.5))
        raise RuntimeError("fake_train: forced crash for harness testing")

    if mode == "timeout":
        # Sleep far longer than any sane per-run timeout; the executor's
        # subprocess timeout is what's expected to kill this.
        time.sleep(10_000)
        return

    time.sleep(sleep_s)
    rng = random.Random(a.seed)
    primary = rng.gauss(mean, std)
    gauc = primary + rng.gauss(0, std / 10)
    ndcg5 = primary - rng.gauss(0, std / 10)

    if mode == "bad_output":
        with open(a.out, "w") as fh:
            fh.write("{not valid json")
    else:
        with open(a.out, "w") as fh:
            json.dump({"primary": primary, "gauc": gauc, "ndcg5": ndcg5, "epochs_run": epochs}, fh)

    # Simulated hidden-test metrics, printed for the executor's quarantine
    # step to intercept -- this line must never reach agent-facing code.
    test_primary = primary + rng.gauss(0, std / 2)
    payload = {"primary": test_primary, "gauc": test_primary, "ndcg5": test_primary}
    print(f"{TEST_METRICS_SENTINEL} {json.dumps(payload)}")


if __name__ == "__main__":
    sys.exit(main())
