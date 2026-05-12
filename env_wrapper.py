"""env_wrapper.py -- wraps the arc_agi EnvironmentWrapper API with observation logging.

Usage:
    env = ArcEnvWrapper("ls20-9607627b")   # game_id includes hash suffix
    obs = env.reset()
    obs, done = env.step("ACTION1")
    obs, done = env.step("ACTION6", x=32, y=16)
    env.close()

Every (action, observation, done) tuple is recorded so the agent can replay
its own trajectory when forming hypotheses. Observations are stored as plain
dicts so they can be serialised to JSON for the LLM prompt.

Episode logs (optional) are written to log_dir as:
    {game_name}_{unix_timestamp_int}.json

Real API facts (confirmed 2026-05-07):
    - Module: arc_agi (pip install arc-agi)
    - Entry: arc_agi.Arcade().make(game_id) -> LocalEnvironmentWrapper
    - game_id format: "ls20-9607627b" (title + hash, from arcade.get_environments())
    - env.reset() -> FrameDataRaw
    - env.step(GameAction, data=None, reasoning=None) -> Optional[FrameDataRaw]
      returns None when the game ends (WIN or GAME_OVER)
    - Done detection: obs is None OR obs.state != GameState.NOT_FINISHED
    - ACTION6 (click): env.step(GameAction.ACTION6, data={"x": int, "y": int})
    - FrameDataRaw fields: game_id, state, levels_completed, win_levels,
        action_input, available_actions (list[int]), frame (list[ndarray 64x64]),
        full_reset, guid
    - GameState: NOT_PLAYED, NOT_FINISHED, WIN, GAME_OVER
    - GameAction: RESET=0, ACTION1=1 .. ACTION7=7
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Action name constants (for agent.py / prompts)
# ---------------------------------------------------------------------------

ACTION_1 = "ACTION1"
ACTION_2 = "ACTION2"
ACTION_3 = "ACTION3"
ACTION_4 = "ACTION4"
ACTION_5 = "ACTION5"
ACTION_6 = "ACTION6"   # requires x, y
ACTION_7 = "ACTION7"

ALL_ACTIONS = [ACTION_1, ACTION_2, ACTION_3, ACTION_4, ACTION_5, ACTION_6, ACTION_7]

# Map action name string -> GameAction enum member (populated lazily on first use)
_ACTION_NAME_TO_ENUM: dict[str, Any] = {}


def _get_game_action(name: str) -> Any:
    """Return the GameAction enum member for an action name like 'ACTION1'."""
    if not _ACTION_NAME_TO_ENUM:
        from arcengine.enums import GameAction
        for member in GameAction:
            _ACTION_NAME_TO_ENUM[member.name] = member
    return _ACTION_NAME_TO_ENUM.get(name)


# Map action integer value -> ACTION_* name string (populated lazily)
_ACTION_VALUE_TO_NAME: dict[int, str] = {}


def available_action_names(available_actions: list[int]) -> list[str]:
    """Convert available_actions int list from FrameDataRaw to ACTION_* name strings.

    GameAction._value2member_map_ uses composite tuple keys so GameAction(int)
    fails. Build the int->name map by iterating the enum once instead.
    """
    if not _ACTION_VALUE_TO_NAME:
        from arcengine.enums import GameAction
        for member in GameAction:
            _ACTION_VALUE_TO_NAME[member.value] = member.name
    return [_ACTION_VALUE_TO_NAME[v] for v in available_actions if v in _ACTION_VALUE_TO_NAME]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Single step in a game episode."""
    step_idx: int
    action: str
    action_data: dict          # {"x": int, "y": int} for ACTION6, else {}
    observation: dict
    done: bool
    levels_completed: int
    elapsed_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class EpisodeLog:
    """Full episode log for one game environment."""
    game_name: str
    started_at: float
    steps: list[StepRecord] = field(default_factory=list)
    final_score: float | None = None
    completed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class ArcEnvWrapper:
    """Wraps arc_agi.LocalEnvironmentWrapper with observation logging.

    Parameters
    ----------
    game_id:
        Full game ID including hash suffix, e.g. "ls20-9607627b".
        Use arcade.get_environments() to list all available IDs.
    log_dir:
        If provided, episode logs are written here as JSON on close().
    arcade:
        Pre-constructed arc_agi.Arcade instance (shared across episodes).
        If None, a new Arcade is created on reset().
    """

    def __init__(
        self,
        game_id: str,
        log_dir: Path | str | None = None,
        arcade: Any = None,
    ) -> None:
        self.game_id = game_id
        self.game_name = game_id.split("-")[0]   # "ls20" from "ls20-9607627b"
        self.log_dir = Path(log_dir) if log_dir else None
        self._arcade = arcade

        self._env: Any = None
        self._episode: EpisodeLog | None = None
        self._step_idx = 0
        self._levels_completed = 0
        self._win_levels = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Instantiate (or re-instantiate) the environment.

        Returns the first observation as a plain dict.
        """
        import arc_agi
        if self._arcade is None:
            self._arcade = arc_agi.Arcade()

        self._env = self._arcade.make(self.game_id)
        self._step_idx = 0
        self._levels_completed = 0
        self._episode = EpisodeLog(
            game_name=self.game_name,
            started_at=time.time(),
        )

        raw_obs = self._env.reset()
        self._grid: list[list[int]] | None = self._capture_grid()
        self._update_level_state(raw_obs)
        return self._normalise_obs(raw_obs)

    def step(self, action: str, x: int | None = None, y: int | None = None) -> tuple[dict, bool]:
        """Submit one action and return (observation, done).

        Parameters
        ----------
        action:
            One of the ACTION_* constants in this module.
        x, y:
            Required only for ACTION6.

        Returns
        -------
        (obs_dict, done)
            obs_dict: normalised observation as plain dict
            done: True when state is WIN or GAME_OVER (or env returns None)
        """
        if self._env is None:
            raise RuntimeError("Call reset() before step()")

        action_enum = _get_game_action(action)
        if action_enum is None:
            raise ValueError(f"Unknown action: {action!r}. Valid: {ALL_ACTIONS}")

        action_data: dict[str, Any] = {}
        if action == ACTION_6:
            if x is None or y is None:
                raise ValueError("ACTION6 requires x and y")
            action_data = {"x": x, "y": y}

        t0 = time.perf_counter()
        try:
            raw_obs = self._env.step(
                action_enum,
                data=action_data if action_data else None,
            )
        except Exception as exc:
            raise RuntimeError(
                f"arc_agi step failed at step {self._step_idx}: "
                f"action={action} data={action_data}"
            ) from exc
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # step() returns None when game ends
        done = raw_obs is None or self._is_done(raw_obs)
        self._grid = self._capture_grid()

        if raw_obs is not None:
            self._update_level_state(raw_obs)
            obs = self._normalise_obs(raw_obs)
        else:
            obs = {
                "game_id": self.game_id,
                "state": "DONE",
                "levels_completed": self._levels_completed,
                "win_levels": self._win_levels,
            }

        record = StepRecord(
            step_idx=self._step_idx,
            action=action,
            action_data=action_data,
            observation=obs,
            done=done,
            levels_completed=self._levels_completed,
            elapsed_ms=elapsed_ms,
        )
        if self._episode is not None:
            self._episode.steps.append(record)

        self._step_idx += 1
        return obs, done

    def close(self, score: float | None = None) -> EpisodeLog | None:
        """Finalise the episode log and optionally persist it to log_dir."""
        if self._episode is None:
            return None

        self._episode.final_score = score
        self._episode.completed = True

        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            ts = int(self._episode.started_at)
            log_path = self.log_dir / f"{self.game_name}_{ts}.json"
            log_path.write_text(self._episode.to_json(), encoding="utf-8")

        episode = self._episode
        self._episode = None
        self._env = None
        return episode

    # ------------------------------------------------------------------
    # Grid capture
    # ------------------------------------------------------------------

    def _capture_grid(self) -> list[list[int]] | None:
        """Capture 32x32 ARC color grid from the underlying game engine."""
        try:
            game = self._env._game
            cam = game.camera
            pixels = game.get_pixels(0, 0, cam.width, cam.height)
            return pixels.tolist()
        except Exception:
            return None

    @property
    def grid_as_text(self) -> str:
        """32x32 ARC grid as a compact text block for LLM prompts."""
        if self._grid is None:
            return "(grid unavailable)"
        return "\n".join(" ".join(str(v) for v in row) for row in self._grid)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _normalise_obs(self, raw_obs: Any) -> dict:
        """Convert FrameDataRaw to a plain serialisable dict for LLM prompts.

        Strips the raw frame array (too large for prompts); preserves all
        metadata fields the agent needs for hypothesis formation.
        """
        try:
            from arcengine.enums import FrameDataRaw
            if not isinstance(raw_obs, FrameDataRaw):
                return raw_obs if isinstance(raw_obs, dict) else {"raw": str(raw_obs)}
        except ImportError:
            pass

        return {
            "game_id": raw_obs.game_id,
            "state": raw_obs.state.value if hasattr(raw_obs.state, "value") else str(raw_obs.state),
            "levels_completed": raw_obs.levels_completed,
            "win_levels": raw_obs.win_levels,
            "available_actions": raw_obs.available_actions,   # list[int] e.g. [1,2,4]
            "full_reset": raw_obs.full_reset,
            "guid": raw_obs.guid,                             # step identifier
        }

    @staticmethod
    def _is_done(raw_obs: Any) -> bool:
        """Return True when the game has ended (WIN or GAME_OVER)."""
        try:
            from arcengine.enums import GameState
            return raw_obs.state in (GameState.WIN, GameState.GAME_OVER)
        except Exception:
            return False

    def _update_level_state(self, raw_obs: Any) -> None:
        if raw_obs is not None and hasattr(raw_obs, "levels_completed"):
            self._levels_completed = raw_obs.levels_completed
        if raw_obs is not None and hasattr(raw_obs, "win_levels"):
            self._win_levels = raw_obs.win_levels

    # ------------------------------------------------------------------
    # Trajectory helpers consumed by agent.py
    # ------------------------------------------------------------------

    def trajectory_summary(self, max_steps: int = 20) -> list[dict]:
        """Last `max_steps` of the episode as compact dicts for LLM prompts."""
        if self._episode is None:
            return []
        return [
            {
                "step": s.step_idx,
                "action": s.action,
                "data": s.action_data,
                "obs": _summarise_obs(s.observation),
                "done": s.done,
                "levels_completed": s.levels_completed,
            }
            for s in self._episode.steps[-max_steps:]
        ]

    @property
    def total_steps(self) -> int:
        return self._step_idx

    @property
    def levels_completed(self) -> int:
        return self._levels_completed

    @property
    def win_levels(self) -> int:
        return self._win_levels


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _summarise_obs(obs: dict, max_len: int = 200) -> str:
    """One-line truncated JSON summary of an observation for LLM prompts."""
    text = json.dumps(obs, default=str)
    return text[:max_len] + "..." if len(text) > max_len else text
