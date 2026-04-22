#!/usr/bin/env python3
"""
Run all agent harness tests.

Usage:
    python tests/run_all.py
"""

import subprocess
import sys
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TESTS = [
    ("test_events.py", "Event System"),
    ("test_sessions.py", "Session & Turn Lifecycle"),
    ("test_tools.py", "Tool Registry"),
    ("test_service.py", "AgentService Orchestration"),
    ("test_live_vllm.py", "Live Integration (vLLM)"),
]


def main():
    print("=" * 60)
    print("  CC-UI Agent Harness — Full Test Suite")
    print("=" * 60)

    results = []
    total_t0 = time.time()

    for filename, label in TESTS:
        path = os.path.join(HERE, filename)
        print(f"\n{'─' * 60}")
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, path],
            cwd=ROOT,
            capture_output=False,
        )
        elapsed = time.time() - t0
        status = "PASS" if result.returncode == 0 else "FAIL"
        results.append((label, status, elapsed))

    total_elapsed = time.time() - total_t0

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for label, status, elapsed in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {label:40s} {status:4s}  ({elapsed:.2f}s)")
    print(f"\n  Total: {sum(1 for _,s,_ in results if s=='PASS')}/{len(results)} passed in {total_elapsed:.2f}s")
    print("=" * 60)

    if any(s == "FAIL" for _, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
