"""agent.py -- hypothesis-test-plan loop agent for ARC-AGI-3.

The agent runs a three-phase loop per episode:

  Phase 1 -- EXPLORE
    Take informative actions to observe environment mechanics.
    After each action, call the LLM to update the world-model hypothesis.
    Continue until hypothesis confidence is HIGH or explore_budget is exhausted.

  Phase 2 -- VERIFY
    Execute 1-3 targeted test actions to confirm or falsify the top hypothesis.
    If hypothesis changes substantially, return to Phase 1.

  Phase 3 -- PLAN + EXECUTE
    Ask the LLM to generate an efficient action plan against the stable world model.
    Execute the plan step-by-step. If a step diverges from predictions, fall back
    to Phase 1 with the new observations.

Efficiency note: the metric is agent_actions / human_median_actions. Every
unnecessary action costs score. The explore budget is kept small; the planner is
instructed to minimise action count.

Called by run_eval.py:
    from agent import ArcAgent
    agent = ArcAgent(backend)
    result = agent.run_episode("ls20")
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any  # noqa: F401 — used in ArcAgent.__init__ arcade param

from env_wrapper import (
    ACTION_1,
    ACTION_6,
    ALL_ACTIONS,
    ArcEnvWrapper,
    EpisodeLog,
)
from llm_backend import LLMBackend, format_hypothesis_prompt, format_plan_prompt


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_EXPLORE_BUDGET = 15   # max actions in EXPLORE before forcing PLAN
DEFAULT_ACTION_BUDGET = 200   # hard cap on total actions per episode
TRAJECTORY_WINDOW = 10        # steps passed to LLM in each prompt
HIGH_CONFIDENCE_TOKENS = {"HIGH", "high", "confident", "certain"}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Outcome of one full episode (returned to run_eval.py)."""
    game_id: str                     # full ID e.g. "ls20-9607627b"
    game_name: str                   # short title e.g. "ls20"
    total_actions: int
    solved: bool
    final_hypothesis: str
    phases_entered: list[str]        # e.g. ["EXPLORE", "VERIFY", "PLAN"]
    episode_log: EpisodeLog | None   # full step log from env_wrapper
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "game_name": self.game_name,
            "total_actions": self.total_actions,
            "solved": self.solved,
            "final_hypothesis": self.final_hypothesis,
            "phases_entered": self.phases_entered,
            "notes": self.notes,
        }

    def __str__(self) -> str:
        status = "SOLVED" if self.solved else "UNSOLVED"
        return (
            f"[{status}] {self.game_name} | "
            f"actions={self.total_actions} | "
            f"phases={self.phases_entered}"
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ArcAgent:
    """Hypothesis-test-plan loop agent for ARC-AGI-3.

    Parameters
    ----------
    backend:
        Initialised LLMBackend (shared across episodes to avoid reload overhead).
    explore_budget:
        Max actions in EXPLORE phase before transitioning to PLAN.
    action_budget:
        Hard cap on total actions per episode.
    log_dir:
        If set, episode logs are written here by ArcEnvWrapper.
    """

    def __init__(
        self,
        backend: LLMBackend,
        explore_budget: int = DEFAULT_EXPLORE_BUDGET,
        action_budget: int = DEFAULT_ACTION_BUDGET,
        log_dir: Path | str | None = None,
        arcade: Any = None,
    ) -> None:
        self.backend = backend
        self.explore_budget = explore_budget
        self.action_budget = action_budget
        self.log_dir = Path(log_dir) if log_dir else None
        self._arcade = arcade   # shared arc_agi.Arcade instance across episodes

    # ------------------------------------------------------------------
    # Entropy helper (used by _explore for paper experiment logging)
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy(confidences: list[float]) -> float:
        """Shannon entropy of a confidence distribution in bits.

        confidences: non-negative weights; need not sum to 1.
        Returns 0.0 for empty or single-element input.
        Used as a proxy for hypothesis uncertainty during EXPLORE phase.
        """
        if len(confidences) <= 1:
            return 0.0
        total = sum(confidences) + 1e-12
        probs = [c / total for c in confidences]
        return -sum(p * math.log2(p + 1e-12) for p in probs)

    # ------------------------------------------------------------------
    # Main entrypoint (called by run_eval.py)
    # ------------------------------------------------------------------

    def run_episode(self, game_id: str) -> EpisodeResult:
        """Run one full episode on a game environment.

        Each environment is seen exactly once at evaluation time — no replays.

        Parameters
        ----------
        game_id:
            Full game ID including hash suffix, e.g. "ls20-9607627b".
        """
        env = ArcEnvWrapper(game_id, log_dir=self.log_dir, arcade=self._arcade)
        game_name = game_id.split("-")[0]
        obs = env.reset()

        hypothesis = ""
        phases_entered: list[str] = []
        notes: list[str] = []
        solved = False

        # Phase 1: EXPLORE
        phases_entered.append("EXPLORE")
        obs, hypothesis, done, explore_notes = self._explore(env, obs, hypothesis)
        notes.extend(explore_notes)
        if done:
            solved = True

        # Phase 2: VERIFY
        if not done and hypothesis:
            phases_entered.append("VERIFY")
            obs, hypothesis, done, verify_notes = self._verify(env, obs, hypothesis)
            notes.extend(verify_notes)
            if done:
                solved = True

        # Phase 3: PLAN + EXECUTE
        if not done:
            phases_entered.append("PLAN")
            obs, done, plan_notes = self._plan_and_execute(env, obs, hypothesis)
            notes.extend(plan_notes)
            if done:
                solved = True

        episode_log = env.close(score=None)

        return EpisodeResult(
            game_id=game_id,
            game_name=game_name,
            total_actions=env.total_steps,
            solved=solved,
            final_hypothesis=hypothesis,
            phases_entered=phases_entered,
            episode_log=episode_log,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Phase 1: EXPLORE
    # ------------------------------------------------------------------

    def _explore(
        self,
        env: ArcEnvWrapper,
        obs: dict,
        hypothesis: str,
    ) -> tuple[dict, str, bool, list[str]]:
        """Spend up to explore_budget actions building a world model.

        Returns (final_obs, updated_hypothesis, done, entropy_notes).
        entropy_notes: per-step H(belief) log strings for paper experiments.
        """
        entropy_notes: list[str] = []

        for _ in range(self.explore_budget):
            if env.total_steps >= self.action_budget:
                break

            trajectory = env.trajectory_summary(TRAJECTORY_WINDOW)
            prompt = format_hypothesis_prompt(
                env.game_name, trajectory, hypothesis, obs,
                grid_text=env.grid_as_text,
            )
            llm_out = self.backend.generate(prompt)
            parsed = _parse_hypothesis_response(llm_out)
            hypothesis = parsed.get("hypothesis", hypothesis)

            # Entropy proxy: uncertain field length inversely proxies confidence.
            # Longer UNCERTAIN text → less confident → higher entropy.
            uncertain_len = len(parsed.get("uncertain", ""))
            conf = 1.0 / (1.0 + uncertain_len / 50.0)
            step_h = self._entropy([conf, 1.0 - conf])
            entropy_notes.append(
                f"EXPLORE step={env.total_steps} H={step_h:.3f}"
            )

            action = parsed.get("next_action", ACTION_1)
            x, y = parsed.get("x"), parsed.get("y")
            obs, done = env.step(action, x=x, y=y)
            if done:
                return obs, hypothesis, True, entropy_notes

            # Transition early if LLM signals high confidence
            if any(tok in parsed.get("confidence", "") for tok in HIGH_CONFIDENCE_TOKENS):
                break

        return obs, hypothesis, False, entropy_notes

    # ------------------------------------------------------------------
    # Phase 2: VERIFY
    # ------------------------------------------------------------------

    def _verify(
        self,
        env: ArcEnvWrapper,
        obs: dict,
        hypothesis: str,
    ) -> tuple[dict, str, bool, list[str]]:
        """Execute targeted test actions to confirm/falsify the hypothesis.

        Returns (final_obs, updated_hypothesis, done, notes).
        """
        notes: list[str] = []
        verify_budget = min(3, self.action_budget - env.total_steps)

        for i in range(verify_budget):
            trajectory = env.trajectory_summary(TRAJECTORY_WINDOW)
            prompt = format_hypothesis_prompt(
                env.game_name, trajectory, hypothesis, obs,
                grid_text=env.grid_as_text,
            )
            llm_out = self.backend.generate(prompt)
            parsed = _parse_hypothesis_response(llm_out)
            updated = parsed.get("hypothesis", hypothesis)

            if updated != hypothesis:
                notes.append(f"Hypothesis revised in verify step {i}")
            hypothesis = updated

            action = parsed.get("next_action", ACTION_1)
            x, y = parsed.get("x"), parsed.get("y")
            obs, done = env.step(action, x=x, y=y)
            if done:
                return obs, hypothesis, True, notes

        return obs, hypothesis, False, notes

    # ------------------------------------------------------------------
    # Phase 3: PLAN + EXECUTE
    # ------------------------------------------------------------------

    def _plan_and_execute(
        self,
        env: ArcEnvWrapper,
        obs: dict,
        hypothesis: str,
    ) -> tuple[dict, bool, list[str]]:
        """Generate a minimal action plan and execute it.

        Returns (final_obs, done, notes).
        """
        notes: list[str] = []
        budget_remaining = self.action_budget - env.total_steps

        prompt = format_plan_prompt(env.game_name, hypothesis, obs, budget_remaining)
        llm_out = self.backend.generate(prompt)
        plan = _parse_plan_response(llm_out)

        if not plan:
            notes.append("LLM returned empty plan; defaulting to ACTION1 sequence")
            plan = [{"action": ACTION_1, "x": None, "y": None}] * min(5, budget_remaining)

        for step in plan:
            if env.total_steps >= self.action_budget:
                notes.append("Action budget exhausted during plan execution")
                break
            action = step.get("action", ACTION_1)
            x, y = step.get("x"), step.get("y")
            obs, done = env.step(action, x=x, y=y)
            if done:
                return obs, True, notes

        return obs, False, notes


# ---------------------------------------------------------------------------
# LLM response parsers
# ---------------------------------------------------------------------------

def _parse_hypothesis_response(text: str) -> dict[str, Any]:
    """Extract fields from the hypothesis-update LLM response.

    Expected format:
        HYPOTHESIS: <updated theory>
        UNCERTAIN: <open questions>
        NEXT_ACTION: <ACTION1|...|ACTION6 x=N y=N|ACTION7>
        REASON: <reasoning>
    """
    result: dict[str, Any] = {
        "hypothesis": "",
        "uncertain": "",
        "next_action": ACTION_1,
        "x": None,
        "y": None,
        "confidence": "",
    }
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("HYPOTHESIS:"):
            result["hypothesis"] = line[len("HYPOTHESIS:"):].strip()
        elif line.startswith("UNCERTAIN:"):
            result["uncertain"] = line[len("UNCERTAIN:"):].strip()
        elif line.startswith("NEXT_ACTION:"):
            action_str = line[len("NEXT_ACTION:"):].strip()
            action, x, y = _parse_action_str(action_str)
            result["next_action"] = action
            result["x"] = x
            result["y"] = y
        elif line.startswith("REASON:"):
            reason = line[len("REASON:"):].strip().upper()
            if any(tok in reason for tok in HIGH_CONFIDENCE_TOKENS):
                result["confidence"] = "HIGH"
    return result


def _parse_plan_response(text: str) -> list[dict[str, Any]]:
    """Extract action list from the plan LLM response.

    Expected format:
        PLAN: ACTION1,ACTION3,ACTION6 x=2 y=4,ACTION2
    """
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("PLAN:"):
            plan_str = line[len("PLAN:"):].strip()
            steps = []
            for token in plan_str.split(","):
                action, x, y = _parse_action_str(token.strip())
                steps.append({"action": action, "x": x, "y": y})
            return steps
    return []


def _parse_action_str(action_str: str) -> tuple[str, int | None, int | None]:
    """Parse "ACTION6 x=2 y=4" or "ACTION1" into (action, x, y)."""
    parts = action_str.strip().split()
    action = parts[0] if parts else ACTION_1
    if action not in ALL_ACTIONS:
        action = ACTION_1

    x: int | None = None
    y: int | None = None
    if action == ACTION_6:
        xm = re.search(r"x=(-?\d+)", action_str)
        ym = re.search(r"y=(-?\d+)", action_str)
        if xm:
            x = int(xm.group(1))
        if ym:
            y = int(ym.group(1))

    return action, x, y
