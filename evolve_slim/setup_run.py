"""
Create a fresh run folder for an autoresearch experiment.

Usage:
    uv run setup_run.py <tag>

Result:
    Creates  evolve_slim/runs/<tag>/  containing:
      src/                 → symlink to ../../src (fixed suite, scoring, best known losses; read-only)
      program.md           → copy of the agent instructions
      slim.py              → fresh local copy: this is the file the agent edits
      results/             → copy of the baseline leaderboard (overall_results.csv, problem_results.csv)
      slim_lib/            → empty, for snapshots of every attempt

The agent then `cd`s into the run folder and works only there: no git branch,
no commits: every change is local to that folder.  Runs live inside
evolve_slim/ so that `uv run` resolves this project's environment.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

SYMLINK_NAMES = ["src"]
COPY_NAMES = ["slim.py", "results", "program.md"]


def setup_run(tag: str, track: str = "") -> str:
    evolve_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(evolve_dir, "runs", tag)
    if os.path.exists(run_dir):
        raise FileExistsError(f"Run folder already exists: {run_dir}")
    os.makedirs(run_dir)

    for name in SYMLINK_NAMES:
        src_path = os.path.join(evolve_dir, name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing source path: {src_path}")
        os.symlink(os.path.relpath(src_path, run_dir), os.path.join(run_dir, name))

    base = os.path.join(evolve_dir, "tracks", track) if track else evolve_dir
    for name in COPY_NAMES:
        src_path = os.path.join(base, name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing source path: {src_path} (run `uv run run_baselines.py` first)")
        if os.path.isdir(src_path):
            shutil.copytree(src_path, os.path.join(run_dir, name),
                            ignore=shutil.ignore_patterns("*.log", "__pycache__", "hidden*", "t[0-9]*"))
        else:
            shutil.copy2(src_path, os.path.join(run_dir, name))

    os.makedirs(os.path.join(run_dir, "slim_lib"))
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag", help="Run tag, e.g. 'sep26-run1'.")
    parser.add_argument("--track", default="", help="take slim.py, program.md and results/ from tracks/<track>/ "
                                                    "(e.g. 'exact' or 'fine') instead of the top level")
    args = parser.parse_args()
    try:
        run_dir = setup_run(args.tag, args.track)
    except (FileExistsError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"Created run folder: {run_dir}")
    print()
    print("Next steps:")
    print(f"  cd {os.path.relpath(run_dir)}")
    print("  uv run slim.py   # one experiment iteration")
