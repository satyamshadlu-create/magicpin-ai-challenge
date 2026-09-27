#!/usr/bin/env python3
"""
generate_submission.py
======================
Generates submission.jsonl — 30 lines covering all test pairs.
Handles free-tier rate limits (15 RPM) with retry + backoff.
Resume-safe: re-run after interruption to pick up where you left off.

Usage:
  export GEMINI_API_KEY=your_key_here
  python generate_submission.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bot import compose, GEMINI_API_KEY

DATASET = Path(__file__).parent / "dataset" / "expanded"
OUT = Path(__file__).parent / "submission.jsonl"
SECONDS_BETWEEN_CALLS = 4.5   # keeps under 15 RPM free-tier limit
MAX_RETRIES = 4


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compose_with_retry(category, merchant, trg, customer):
    for attempt in range(MAX_RETRIES):
        try:
            return compose(category, merchant, trg, customer)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                m = re.search(r"retry in (\d+)", msg)
                wait = int(m.group(1)) + 5 if m else 65
                print(f"\n  [429] rate-limited — waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...",
                      end="", flush=True)
                time.sleep(wait)
            else:
                print(f"\n  [ERROR] {msg[:120]}")
                return None
    print(f"\n  [FAIL] max retries reached")
    return None


def main():
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set. Set it and re-run.")
        sys.exit(1)

    pairs_path = DATASET / "test_pairs.json"
    if not pairs_path.exists():
        print(f"ERROR: {pairs_path} not found. Run: python dataset/generate_dataset.py --seed-dir dataset --out dataset/expanded")
        sys.exit(1)

    pairs = load_json(pairs_path)["pairs"]
    print(f"Found {len(pairs)} test pairs")

    # Load contexts
    categories = {d["slug"]: d for f in (DATASET / "categories").glob("*.json") for d in [load_json(f)]}
    merchants  = {d["merchant_id"]: d for f in (DATASET / "merchants").glob("*.json") for d in [load_json(f)]}
    customers  = {d["customer_id"]: d for f in (DATASET / "customers").glob("*.json") for d in [load_json(f)]}
    triggers   = {d["id"]: d for f in (DATASET / "triggers").glob("*.json") for d in [load_json(f)]}
    print(f"Loaded: {len(categories)} categories, {len(merchants)} merchants, "
          f"{len(customers)} customers, {len(triggers)} triggers")

    # Resume: load already-written test_ids
    done: dict[str, str] = {}   # test_id -> json line
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done[json.loads(line)["test_id"]] = line
                    except Exception:
                        pass
    if done:
        print(f"Resuming: {len(done)} already done: {', '.join(sorted(done))}")

    results: dict[str, str] = dict(done)

    for pair in pairs:
        test_id     = pair["test_id"]
        trigger_id  = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")

        if test_id in results:
            print(f"Skipping {test_id} (already done)")
            continue

        print(f"Composing {test_id}: {trigger_id} / {merchant_id}...", end="", flush=True)

        trg      = triggers.get(trigger_id)
        merchant = merchants.get(merchant_id)
        if not trg or not merchant:
            print(" SKIP (missing context)")
            continue
        category = categories.get(merchant.get("category_slug", ""))
        if not category:
            print(f" SKIP (missing category)")
            continue
        customer = customers.get(customer_id) if customer_id else None

        result = compose_with_retry(category, merchant, trg, customer)
        if result is None:
            print(" FAILED — skipping")
            continue

        line_obj = {
            "test_id":        test_id,
            "body":           result.get("body", ""),
            "cta":            result.get("cta", "open_ended"),
            "send_as":        result.get("send_as", "vera"),
            "suppression_key": result.get("suppression_key", ""),
            "rationale":      result.get("rationale", ""),
        }
        results[test_id] = json.dumps(line_obj, ensure_ascii=False)
        print(f" OK ({len(line_obj['body'])} chars)")

        # Write incrementally (safe for interruption)
        with open(OUT, "w", encoding="utf-8") as f:
            for tid in sorted(results):
                f.write(results[tid] + "\n")

        time.sleep(SECONDS_BETWEEN_CALLS)

    total = len(results)
    print(f"\nDone. {total}/30 pairs written to {OUT}")
    if total < 30:
        print("Re-run this script to resume remaining pairs (rate-limit window resets after 1 min).")


if __name__ == "__main__":
    main()
