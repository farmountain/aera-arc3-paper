"""run_eval.py -- evaluate the ARC-AGI-3 agent across all public game environments.

Usage (random baseline, all public games):
    python competitions/arc-agi-3/run_eval.py \
        --games public \
        --output runs/scorecard_baseline.json

Usage (local dev, specific games with LLM):
    python competitions/arc-agi-3/run_eval.py \
        --model /path/to/model.gguf \
        --games ls20 ft09

Usage (all public games with LLM, sorted easiest-first):
    python competitions/arc-agi-3/run_eval.py \
        --model /path/to/model.gguf \
        --all

Kaggle notebook: set MODEL_PATH then call main() directly.

Scorecard written to --output path (or runs/arc-agi-3/{run_id}/scorecard.json):
    {
        "total_score": 0.023,
        "games": {
            "ls20": {"score": 0.05, "failure_mode": "fixation",
                     "actions_used": 30, "actions_budget": 50},
            "ft09": {"score": 0.0, "failure_mode": "wrong_hypothesis",
                     "actions_used": 50, "actions_budget": 50}
        }
    }
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running from competitions/arc-agi-3/ or from repo root
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from agent import ArcAgent, EpisodeResult
from llm_backend import BACKEND_ANTHROPIC, BACKEND_LLAMACPP, BACKEND_AIRLLM, BACKEND_STUB, LLMBackend


# ---------------------------------------------------------------------------
# Public game list (all 25, confirmed 2026-05-07 via arcade.get_environments())
# Sorted by sum(baseline_actions) ascending — easiest games first.
# Playing easy games first banks efficiency score early.
# ---------------------------------------------------------------------------

PUBLIC_GAMES: list[dict] = [
    # id                        title    tags                   baseline_actions (per level)
    {"id": "sb26-7fbdac44",  "title": "SB26",  "tags": ["keyboard_click"], "baseline": [18,28,18,19,31,23,58,18]},
    {"id": "ft09-0d8bbf25",  "title": "FT09",  "tags": [],                 "baseline": [43,12,23,28,65,37]},
    {"id": "cd82-fb555c5d",  "title": "CD82",  "tags": ["keyboard_click"], "baseline": [55,8,41,21,23,23]},
    {"id": "tu93-0768757b",  "title": "TU93",  "tags": ["keyboard_click"], "baseline": [19,16,34,42,123,80,14,23,111]},
    {"id": "r11l-495a7899",  "title": "R11L",  "tags": ["click"],          "baseline": [22,33,51,26,52,49]},
    {"id": "sc25-635fd71a",  "title": "SC25",  "tags": ["keyboard_click"], "baseline": [36,6,32,83,143,50]},
    {"id": "lp85-305b61c3",  "title": "LP85",  "tags": ["click"],          "baseline": [17,38,31,16,41,60,26,159]},
    {"id": "tn36-ef4dde99",  "title": "TN36",  "tags": ["click"],          "baseline": [32,72,26,40,30,55,62]},
    {"id": "su15-1944f8ab",  "title": "SU15",  "tags": ["click"],          "baseline": [22,42,26,115,36,31,8,40,41]},
    {"id": "vc33-5430563c",  "title": "VC33",  "tags": ["click"],          "baseline": [7,18,44,61,131,34,152]},
    {"id": "sp80-589a99af",  "title": "SP80",  "tags": ["keyboard_click"], "baseline": [39,58,25,148,96,152]},
    {"id": "tr87-cd924810",  "title": "TR87",  "tags": ["keyboard"],       "baseline": [54,58,40,45,71,146]},
    {"id": "cn04-2fe56bfb",  "title": "CN04",  "tags": ["keyboard_click"], "baseline": [29,54,85,300,208,113]},
    {"id": "ar25-0c556536",  "title": "AR25",  "tags": ["keyboard_click"], "baseline": [32,50,75,37,89,159,233,73]},
    {"id": "ka59-38d34dbb",  "title": "KA59",  "tags": ["keyboard_click"], "baseline": [28,109,51,51,33,132,326]},
    {"id": "sk48-d8078629",  "title": "SK48",  "tags": ["keyboard_click"], "baseline": [61,177,101,103,230,181,125,92]},
    {"id": "s5i5-18d95033",  "title": "S5I5",  "tags": ["click"],          "baseline": [20,89,106,54,162,38,86,83]},
    {"id": "bp35-0a0ad940",  "title": "BP35",  "tags": ["keyboard_click"], "baseline": [21,48,44,38,33,87,86,131,163]},
    {"id": "g50t-5849a774",  "title": "G50T",  "tags": ["keyboard"],       "baseline": [78,175,179,230,96,54,67]},
    {"id": "dc22-fdcac232",  "title": "DC22",  "tags": ["keyboard_click"], "baseline": [59,102,67,98,324,578]},
    {"id": "re86-8af5384d",  "title": "RE86",  "tags": ["keyboard_click"], "baseline": [26,42,86,108,189,139,424,241]},
    {"id": "ls20-9607627b",  "title": "LS20",  "tags": ["keyboard"],       "baseline": [22,123,73,84,96,192,186]},
    {"id": "m0r0-492f87ba",  "title": "M0R0",  "tags": ["keyboard_click"], "baseline": [30,111,203,26,500,237]},
    {"id": "lf52-271a04aa",  "title": "LF52",  "tags": ["click"],          "baseline": [32,81,60,71,205,148,244,109,164,225]},
    {"id": "wa30-ee6fef47",  "title": "WA30",  "tags": ["keyboard"],       "baseline": [71,119,183,98,368,68,79,442,415]},
]

_GAME_BY_TITLE: dict[str, dict] = {g["title"].lower(): g for g in PUBLIC_GAMES}
_GAME_BY_ID: dict[str, dict] = {g["id"]: g for g in PUBLIC_GAMES}


def resolve_game_ids(names: list[str]) -> list[str]:
    """Resolve 'ls20' or 'ls20-9607627b' to full game IDs."""
    result = []
    for name in names:
        if name in _GAME_BY_ID:
            result.append(name)
        elif name.lower() in _GAME_BY_TITLE:
            result.append(_GAME_BY_TITLE[name.lower()]["id"])
        else:
            print(f"WARNING: Unknown game {name!r} — skipping", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# Scoring (real competition formula)
# ---------------------------------------------------------------------------

def _per_game_efficiency(result: EpisodeResult, baseline: list[int]) -> float:
    """ARC-AGI-3 official RHAE formula (confirmed from competition rules page):

    per_level = min(human_actions / agent_actions, 1.0)^2
    per_game = weighted_avg(level_scores, weights=[1, 2, 3, ...N])

    Cap is 1.0 (NOT 1.15 — that was wrong). Weighted by level index.
    Agent actions distributed evenly across completed levels as proxy.
    """
    if not result.solved or result.total_actions == 0 or not baseline:
        return 0.0
    n_levels = len(baseline)
    agent_per_level = result.total_actions / n_levels
    weights = list(range(1, n_levels + 1))
    level_scores = [min(h / agent_per_level, 1.0) ** 2 for h in baseline]
    return sum(w * s for w, s in zip(weights, level_scores)) / sum(weights)


def _compute_efficiency_score(results: list[EpisodeResult]) -> float:
    if not results:
        return 0.0
    scores = []
    for r in results:
        game = _GAME_BY_ID.get(r.game_id) or _GAME_BY_TITLE.get(r.game_name.lower(), {})
        scores.append(_per_game_efficiency(r, game.get("baseline", [])))
    return sum(scores) / len(scores)


def _build_scorecard(run_id: str, results: list[EpisodeResult], action_budget: int) -> dict:
    """Build scorecard in the format expected by loop.sh and value_function.py.

    Primary format (used by loop.sh):
        {"total_score": float, "games": {game_name: {score, failure_mode, ...}}}

    Also includes legacy fields for backwards compatibility with kaggle_notebook.py.
    """
    games: dict[str, dict] = {}
    for r in results:
        # Determine failure mode
        if any("Exception" in n or "crash" in n.lower() for n in r.notes):
            failure_mode = "crash"
        elif r.solved:
            failure_mode = "none"
        elif r.total_actions >= action_budget:
            failure_mode = "budget_exhausted"
        else:
            failure_mode = "wrong_hypothesis"

        game = _GAME_BY_ID.get(r.game_id, {})
        eff = _per_game_efficiency(r, game.get("baseline", []))
        games[r.game_name] = {
            "score": round(eff, 6),
            "failure_mode": failure_mode,
            "actions_used": r.total_actions,
            "actions_budget": action_budget,
        }

    total_score = _compute_efficiency_score(results)
    return {
        # Primary keys (loop.sh, value_function.py)
        "total_score": round(total_score, 6),
        "games": games,
        # Legacy keys (kaggle_notebook.py, old readers)
        "run_id": run_id,
        "games_attempted": len(results),
        "games_solved": sum(1 for r in results if r.solved),
        "efficiency_score": round(total_score, 6),
    }


# ---------------------------------------------------------------------------
# Random agent baseline
# ---------------------------------------------------------------------------

def _run_random_episode(arcade: Any, game_id: str, action_budget: int) -> EpisodeResult:
    """Run a random action agent on one game.  Used for baseline scorecards.

    Picks uniformly from available_actions each step.  ACTION6 (click) gets
    random x/y in [0, 63].  Returns when done or budget exhausted.
    """
    from arcengine.enums import GameAction, GameState

    game_name = game_id.split("-")[0]
    try:
        env = arcade.make(game_id)
        obs = env.reset()
        steps = 0
        solved = False

        _val_to_action = {a.value: a for a in GameAction}
        while steps < action_budget:
            avail = obs.available_actions if obs.available_actions else [1]
            action_val = random.choice(avail)
            action_enum = _val_to_action.get(action_val, GameAction.ACTION1)

            data: dict = {}
            if action_enum == GameAction.ACTION6:
                data = {"x": random.randint(0, 63), "y": random.randint(0, 63)}

            result = env.step(action_enum, data=data if data else None)
            steps += 1

            if result is None:
                solved = True
                break
            obs = result
            if obs.state in (GameState.WIN, GameState.GAME_OVER):
                solved = (obs.state == GameState.WIN)
                break

        return EpisodeResult(
            game_id=game_id,
            game_name=game_name,
            total_actions=steps,
            solved=solved,
            final_hypothesis="random",
            phases_entered=["RANDOM"],
            episode_log=None,
            notes=[],
        )
    except Exception as exc:
        return EpisodeResult(
            game_id=game_id,
            game_name=game_name,
            total_actions=0,
            solved=False,
            final_hypothesis="",
            phases_entered=[],
            episode_log=None,
            notes=[f"Exception: {exc}"],
        )


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(
    game_ids: list[str],
    out_dir: Path,
    output_path: Path | None = None,
    model_path: str | Path | None = None,
    backend_type: str = BACKEND_STUB,
    action_budget: int = 200,
    explore_budget: int | None = None,
    n_gpu_layers: int = -1,
    no_explore: bool = False,
    random_baseline: bool = False,
) -> dict:
    """Run the agent on each game and return the scorecard dict.

    When model_path is None or random_baseline=True, uses a random action agent
    (no LLM required) to produce a baseline scorecard immediately.

    explore_budget is set dynamically per game from baseline_actions[0] when
    not explicitly provided.  A single Arcade instance is shared across all
    episodes to avoid re-fetching the anonymous API key on each game.
    """
    import arc_agi
    arcade = arc_agi.Arcade()

    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "episode_logs"
    llm_log = run_dir / "llm_calls.jsonl"

    use_random = random_baseline or (model_path is None)
    agent_label = "RANDOM" if use_random else str(model_path)

    print(f"Run ID : {run_id}")
    print(f"Agent  : {agent_label}")
    print(f"Games  : {len(game_ids)}")
    print(f"Budget : {action_budget} actions/game")
    print(f"Output : {output_path or run_dir / 'scorecard.json'}")
    print()

    backend: LLMBackend | None = None
    if not use_random:
        backend = LLMBackend(
            model_path=model_path,  # type: ignore[arg-type]
            backend=backend_type,
            n_gpu_layers=n_gpu_layers,
            log_path=llm_log,
        )

    results: list[EpisodeResult] = []
    for i, game_id in enumerate(game_ids, 1):
        game_info = _GAME_BY_ID.get(game_id, {})
        baseline = game_info.get("baseline", [])
        title = game_info.get("title", game_id)
        tags = ",".join(game_info.get("tags", [])) or "no-tag"

        print(
            f"[{i}/{len(game_ids)}] {title} ({tags}) "
            f"baseline_sum={sum(baseline)} ...",
            end=" ", flush=True,
        )
        t0 = time.perf_counter()

        if use_random:
            result = _run_random_episode(arcade, game_id, action_budget)
        else:
            # Dynamic explore budget: caller override, or 40% of level-1 baseline
            if explore_budget is not None:
                ep_explore = explore_budget
            elif no_explore:
                ep_explore = 0
            else:
                ep_explore = max(5, min(30, int(0.4 * baseline[0]))) if baseline else 15

            agent = ArcAgent(
                backend=backend,  # type: ignore[arg-type]
                explore_budget=ep_explore,
                action_budget=action_budget,
                log_dir=log_dir,
                arcade=arcade,
            )
            try:
                result = agent.run_episode(game_id)
            except Exception as exc:
                print(f"ERROR: {exc}")
                result = EpisodeResult(
                    game_id=game_id,
                    game_name=game_id.split("-")[0],
                    total_actions=0,
                    solved=False,
                    final_hypothesis="",
                    phases_entered=[],
                    episode_log=None,
                    notes=[f"Exception: {exc}"],
                )

        elapsed = time.perf_counter() - t0
        results.append(result)
        status = "SOLVED" if result.solved else "unsolved"
        print(f"{status} | actions={result.total_actions} | {elapsed:.1f}s")

    scorecard = _build_scorecard(run_id, results, action_budget)

    # Write to explicit --output path if given, else default run dir
    scorecard_path = output_path if output_path is not None else run_dir / "scorecard.json"
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard_path.write_text(
        json.dumps(scorecard, indent=2, default=str), encoding="utf-8"
    )

    print()
    print(f"Scorecard  : {scorecard_path}")
    print(f"Solved     : {scorecard['games_solved']}/{scorecard['games_attempted']}")
    print(f"Score      : {scorecard['total_score']:.6f}")

    return scorecard


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARC-AGI-3 agent evaluation")
    p.add_argument("--model", default=None,
                   help="Path to GGUF or HuggingFace model dir. "
                        "Omit to run a random-action baseline (no LLM required).")
    p.add_argument("--games", nargs="+", default=[],
                   help="Game titles or IDs (e.g. ls20 ft09), or 'public' for all 25. "
                        "Ignored if --all.")
    p.add_argument("--all", action="store_true", dest="all_games",
                   help="Evaluate all 25 public games sorted easiest-first")
    p.add_argument("--output", default=None,
                   help="Direct path for scorecard JSON (e.g. runs/scorecard_baseline.json). "
                        "If omitted, writes to runs/arc-agi-3/{run_id}/scorecard.json.")
    p.add_argument("--out-dir", default="runs/arc-agi-3",
                   help="Output directory for run artifacts (default: runs/arc-agi-3). "
                        "Ignored when --output is set.")
    p.add_argument("--backend", default=BACKEND_STUB,
                   choices=["llamacpp", "vllm", "airllm", "stub", "anthropic", "transformers"])
    p.add_argument("--action-budget", type=int, default=200,
                   help="Hard cap on total actions per episode (default: 200)")
    p.add_argument("--explore-budget", type=int, default=None,
                   help="Override explore phase budget (default: auto from baseline)")
    p.add_argument("--n-gpu-layers", type=int, default=-1,
                   help="GPU layers for llama.cpp (-1=all, 0=CPU-only)")
    p.add_argument("--no-explore", action="store_true",
                   help="Ablation: skip EXPLORE phase (explore_budget=0) — paper experiment")
    p.add_argument("--random", action="store_true", dest="random_baseline",
                   help="Force random-action baseline even if --model is provided")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Resolve game list — 'public' keyword means all 25 public games
    if args.all_games:
        game_ids = [g["id"] for g in PUBLIC_GAMES]
    elif len(args.games) == 1 and args.games[0].lower() == "public":
        game_ids = [g["id"] for g in PUBLIC_GAMES]
    else:
        game_ids = resolve_game_ids(args.games)

    if not game_ids:
        print("No games specified. Use --games public, --games ls20 ft09, or --all",
              file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / args.out_dir
    output_path = Path(args.output) if args.output else None
    # Resolve relative output paths against repo root
    if output_path is not None and not output_path.is_absolute():
        output_path = repo_root / output_path

    run_eval(
        game_ids=game_ids,
        out_dir=out_dir,
        output_path=output_path,
        model_path=args.model,
        backend_type=args.backend,
        action_budget=args.action_budget,
        explore_budget=args.explore_budget,
        n_gpu_layers=args.n_gpu_layers,
        no_explore=args.no_explore,
        random_baseline=args.random_baseline,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
