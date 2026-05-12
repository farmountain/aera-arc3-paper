"""arc_agi3_v4_clean.py - AERA paper experiments EXP-001 and EXP-002.

Setup: New Kaggle Script, GPU P100, Internet ON.
Runtime: ~90 min (Qwen2.5-0.5B CPU FP32, 5 games x 2 experiments).
"""
import subprocess, sys, json, math, re, time
from pathlib import Path
from dataclasses import dataclass, field


def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip("arc-agi==0.9.8")
pip("transformers>=4.40.0", "accelerate>=0.30.0")
print("Dependencies OK")


# === Write agent source files =================================================

ADIR = Path("/kaggle/working/agent")
ADIR.mkdir(exist_ok=True)
(ADIR / "__init__.py").write_text("")

EW = """
from __future__ import annotations
import json, time
from dataclasses import dataclass, field
from pathlib import Path

ACTION_1="ACTION1"; ACTION_2="ACTION2"; ACTION_3="ACTION3"
ACTION_4="ACTION4"; ACTION_5="ACTION5"; ACTION_6="ACTION6"; ACTION_7="ACTION7"
ALL_ACTIONS=[ACTION_1,ACTION_2,ACTION_3,ACTION_4,ACTION_5,ACTION_6,ACTION_7]

_A2E: dict = {}
_V2N: dict = {}

def _get_action(name):
    if not _A2E:
        from arcengine.enums import GameAction
        for m in GameAction: _A2E[m.name] = m
    return _A2E.get(name)

def available_action_names(avail):
    if not _V2N:
        from arcengine.enums import GameAction
        for m in GameAction: _V2N[m.value] = m.name
    return [_V2N[v] for v in (avail or []) if v in _V2N]

@dataclass
class StepRecord:
    step_idx: int; action: str; action_data: dict
    observation: dict; done: bool; levels_completed: int; elapsed_ms: float

@dataclass
class EpisodeLog:
    game_id: str; started_at: float
    steps: list = field(default_factory=list)
    completed: bool = False; final_score: object = None
    def to_json(self, indent=2): return json.dumps(self.__dict__, indent=indent, default=str)

def _trim(obs, n=200):
    t = json.dumps(obs, default=str)
    return t[:n]+"..." if len(t)>n else t

class ArcEnvWrapper:
    def __init__(self, game_id, log_dir=None, arcade=None):
        self.game_id=game_id; self.game_name=game_id.split("-")[0]
        self.log_dir=Path(log_dir) if log_dir else None; self._arcade=arcade
        self._env=None; self._episode=None
        self._step_idx=0; self._levels_completed=0; self._win_levels=0

    def reset(self):
        if self._arcade is None:
            import arc_agi; self._arcade=arc_agi.Arcade()
        self._env=self._arcade.make(self.game_id); raw=self._env.reset()
        self._step_idx=0; self._levels_completed=0; self._win_levels=0
        self._episode=EpisodeLog(game_id=self.game_id, started_at=time.time())
        return self._norm(raw) if raw is not None else {"state":"READY"}

    def step(self, action, x=None, y=None):
        ae=_get_action(action)
        if ae is None: raise ValueError(f"Unknown action {action!r}")
        data={"x":x,"y":y} if action==ACTION_6 else {}
        t0=time.perf_counter()
        raw=self._env.step(ae, data=data if data else None)
        ms=(time.perf_counter()-t0)*1000
        done=raw is None or self._is_done(raw)
        obs=self._norm(raw) if raw is not None else {"state":"DONE","levels_completed":self._levels_completed}
        if raw is not None: self._upd(raw)
        if self._episode:
            self._episode.steps.append(StepRecord(self._step_idx,action,data,obs,done,self._levels_completed,ms))
        self._step_idx+=1; return obs,done

    def close(self, score=None):
        if self._episode is None: return None
        self._episode.final_score=score; self._episode.completed=True
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir/f"{self.game_name}_{int(self._episode.started_at)}.json").write_text(self._episode.to_json(),encoding="utf-8")
        ep=self._episode; self._episode=None; self._env=None; return ep

    def trajectory_summary(self, n=5):
        if not self._episode: return []
        return [{"step":s.step_idx,"action":s.action,"obs":_trim(s.observation),"done":s.done,"levels":s.levels_completed}
                for s in self._episode.steps[-n:]]

    @property
    def total_steps(self): return self._step_idx
    @property
    def levels_completed(self): return self._levels_completed

    def _norm(self, raw):
        try:
            from arcengine.enums import FrameDataRaw
            if not isinstance(raw, FrameDataRaw): return raw if isinstance(raw,dict) else {"raw":str(raw)}
        except ImportError: pass
        return {"state":raw.state.value if hasattr(raw.state,"value") else str(raw.state),
                "levels_completed":raw.levels_completed,"win_levels":raw.win_levels,
                "available_actions":available_action_names(raw.available_actions)}

    @staticmethod
    def _is_done(raw):
        try:
            from arcengine.enums import GameState
            return raw.state in (GameState.WIN, GameState.GAME_OVER)
        except: return False

    def _upd(self, raw):
        if hasattr(raw,"levels_completed"): self._levels_completed=raw.levels_completed
        if hasattr(raw,"win_levels"): self._win_levels=raw.win_levels
""".strip()
(ADIR / "env_wrapper.py").write_text(EW)

LLM = """
import json


def _hyp(game, traj, hyp, obs):
    lines = [f"  step {s['step']}: {s['action']}" for s in traj]
    return (
        f'Explore game "{game}" with no prior knowledge.\\n'
        f'Trajectory:\\n' + ('\\n'.join(lines) or '(none)') + '\\n'
        f'Obs: {json.dumps(obs, default=str)[:250]}\\n'
        f'Hypothesis: {hyp or "(none)"}\\n'
        'Reply EXACTLY:\\n'
        'HYPOTHESIS: <theory>\\n'
        'UNCERTAIN: <unknowns>\\n'
        'NEXT_ACTION: <ACTION1..ACTION7>\\n'
        'REASON: <why>'
    )


def _plan(game, hyp, obs, budget):
    return (
        f'Rules "{game}": {hyp}\\n'
        f'State: {json.dumps(obs, default=str)[:200]}\\n'
        f'Budget: {budget} actions.\\n'
        'PLAN: <comma actions>\\n'
        'CONFIDENCE: HIGH\\n'
        'FALLBACK: ACTION1'
    )


class LLMBackend:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", max_tokens=256, temperature=0.2):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model = None
        self._tok = None

    def generate(self, prompt, max_tokens=None):
        if self._model is None:
            self._load()
        import torch
        n = max_tokens or self.max_tokens
        msgs = [{"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self._tok([text], return_tensors="pt")
        with torch.no_grad():
            out = self._model.generate(
                **inp, max_new_tokens=n,
                temperature=self.temperature, do_sample=self.temperature > 0,
            )
        new_tok = out[0][inp.input_ids.shape[-1]:]
        return self._tok.decode(new_tok, skip_special_tokens=True).strip()

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(f"Loading {self.model_name} on CPU (FP32)...")
        self._tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        print("Model loaded OK")
""".strip()
(ADIR / "llm_backend.py").write_text(LLM)

AG = """
from __future__ import annotations
import math, re
from dataclasses import dataclass, field
from env_wrapper import ArcEnvWrapper, ACTION_1
from llm_backend import LLMBackend, _hyp, _plan

HIGH_CONF = {"HIGH", "high", "confident", "certain"}


@dataclass
class EpisodeResult:
    game_id: str; game_name: str; total_actions: int; solved: bool
    final_hypothesis: str; phases_entered: list; episode_log: object
    notes: list = field(default_factory=list)
    def to_dict(self): return {k: v for k, v in self.__dict__.items() if k != "episode_log"}


def _parse(text):
    d = {}
    for k in ("HYPOTHESIS", "UNCERTAIN", "NEXT_ACTION", "PLAN", "CONFIDENCE"):
        m = re.search(rf"^{k}:\\s*(.+)$", text, re.M | re.I)
        if m:
            d[k.lower()] = m.group(1).strip()
    if "next_action" in d:
        parts = d["next_action"].split()
        d["action"] = parts[0] if parts else ACTION_1
        for p in parts[1:]:
            if p.startswith("x="): d["x"] = int(p[2:])
            elif p.startswith("y="): d["y"] = int(p[2:])
    return d


def _H(c):
    if len(c) <= 1: return 0.0
    t = sum(c) + 1e-12
    ps = [x / t for x in c]
    return -sum(p * math.log2(p + 1e-12) for p in ps)


class ArcAgent:
    def __init__(self, backend, explore_budget=5, action_budget=50, log_dir=None, arcade=None):
        self.backend = backend; self.explore_budget = explore_budget
        self.action_budget = action_budget; self.log_dir = log_dir; self._arcade = arcade

    def run_episode(self, game_id):
        gname = game_id.split("-")[0]
        env = ArcEnvWrapper(game_id, log_dir=self.log_dir, arcade=self._arcade)
        obs = env.reset(); hyp = ""; phases = []; notes = []; solved = False

        phases.append("EXPLORE")
        obs, hyp, done, en = self._explore(env, obs, hyp)
        notes.extend(en)
        if done: solved = True

        if not done and hyp:
            phases.append("PLAN")
            obs, done, pn = self._plan_phase(env, obs, hyp)
            notes.extend(pn)
            if done: solved = True

        log = env.close()
        return EpisodeResult(game_id, gname, env.total_steps, solved, hyp, phases, log, notes)

    def _explore(self, env, obs, hyp):
        notes = []
        for _ in range(self.explore_budget):
            if env.total_steps >= self.action_budget: break
            p = _hyp(env.game_name, env.trajectory_summary(5), hyp, obs)
            r = self.backend.generate(p); d = _parse(r)
            hyp = d.get("hypothesis", hyp)
            unc = len(d.get("uncertain", ""))
            conf = 1.0 / (1.0 + unc / 50.0)
            h = _H([conf, 1.0 - conf])
            notes.append(f"EXPLORE step={env.total_steps} H={h:.3f}")
            act = d.get("action", ACTION_1); x, y = d.get("x"), d.get("y")
            obs, done = env.step(act, x=x, y=y)
            if done: return obs, hyp, True, notes
            if any(t in d.get("confidence", "") for t in HIGH_CONF): break
        return obs, hyp, False, notes

    def _plan_phase(self, env, obs, hyp):
        notes = []
        budget = max(1, self.action_budget - env.total_steps)
        p = _plan(env.game_name, hyp, obs, budget)
        r = self.backend.generate(p); d = _parse(r)
        acts = [a.strip() for a in d.get("plan", "").split(",") if a.strip()]
        for astr in acts:
            if env.total_steps >= self.action_budget: break
            parts = astr.split(); act = parts[0] if parts else ACTION_1
            x = y = None
            for pp in parts[1:]:
                if pp.startswith("x="): x = int(pp[2:])
                elif pp.startswith("y="): y = int(pp[2:])
            obs, done = env.step(act, x=x, y=y)
            notes.append(f"PLAN step={env.total_steps} {act}")
            if done: return obs, True, notes
        return obs, False, notes
""".strip()
(ADIR / "agent.py").write_text(AG)
print("Agent files OK")
sys.path.insert(0, str(ADIR))


# === Game catalogue ===========================================================

GAMES = [
    {"id": "sb26-7fbdac44", "title": "SB26", "baseline": [18,28,18,19,31,23,58,18]},
    {"id": "ft09-0d8bbf25", "title": "FT09", "baseline": [43,12,23,28,65,37]},
    {"id": "cd82-fb555c5d", "title": "CD82", "baseline": [55,8,41,21,23,23]},
    {"id": "tu93-0768757b", "title": "TU93", "baseline": [19,16,34,42,123,80,14,23,111]},
    {"id": "r11l-495a7899", "title": "R11L", "baseline": [22,33,51,26,52,49]},
]


def rhae(baseline, ai_actions, solved):
    if not solved or ai_actions == 0 or not baseline: return 0.0
    per = ai_actions / len(baseline)
    return sum(min(h / per, 1.15) ** 2 for h in baseline) / len(baseline)


# === Load model ===============================================================

from llm_backend import LLMBackend  # type: ignore
backend = LLMBackend("Qwen/Qwen2.5-0.5B-Instruct", max_tokens=256, temperature=0.2)
backend._load()


# === Run experiments ==========================================================

import arc_agi
from agent import ArcAgent  # type: ignore


def run_experiment(name, no_explore, out_dir):
    arcade = arc_agi.Arcade()
    out = Path(f"/kaggle/working/runs/{out_dir}")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for g in GAMES:
        gid, title, baseline = g["id"], g["title"], g["baseline"]
        budget = 0 if no_explore else max(2, min(5, int(0.2 * baseline[0])))
        print(f"\n[{name}] {title}  budget={budget}")
        agent = ArcAgent(backend=backend, explore_budget=budget, action_budget=50,
                         log_dir=out/"episodes", arcade=arcade)
        result = agent.run_episode(gid)
        score = rhae(baseline, result.total_actions, result.solved)
        r = result.to_dict(); r["rhae"] = round(score, 4); r["budget"] = budget
        results.append(r)
        status = "SOLVED" if result.solved else "unsolved"
        print(f"  {status}  actions={result.total_actions}  RHAE={score:.4f}  phases={result.phases_entered}")
        for n in result.notes:
            if "H=" in n: print(f"    {n}")
    eff = sum(r["rhae"] for r in results) / len(results)
    sc = {"experiment": name, "no_explore": no_explore,
          "games_attempted": len(results), "games_solved": sum(1 for r in results if r["solved"]),
          "efficiency_score": round(eff, 6), "results": results}
    (out / "scorecard.json").write_text(json.dumps(sc, indent=2))
    print(f"\n[{name}]  RHAE={eff:.4f}  solved={sc['games_solved']}/{len(results)}")
    return sc


print("=" * 50 + "\nEXP-001: no-explore baseline\n" + "=" * 50)
exp001 = run_experiment("EXP-001", no_explore=True, out_dir="exp001")

print("\n" + "=" * 50 + "\nEXP-002: AERA with explore\n" + "=" * 50)
exp002 = run_experiment("EXP-002", no_explore=False, out_dir="exp002")


# === Results ==================================================================

delta = exp002["efficiency_score"] - exp001["efficiency_score"]
print("\nPAPER TABLE 1")
print(f"  B1 no-explore: RHAE={exp001['efficiency_score']:.4f}  solved={exp001['games_solved']}/5")
print(f"  AERA explore:  RHAE={exp002['efficiency_score']:.4f}  solved={exp002['games_solved']}/5")
print(f"  Delta: {delta:+.4f}")
print("PRIMARY CLAIM CONFIRMED" if delta > 0 else "NOT CONFIRMED")

summary = {"exp001_rhae": exp001["efficiency_score"], "exp002_rhae": exp002["efficiency_score"],
           "delta": round(delta, 6), "primary_claim_confirmed": delta > 0,
           "exp001_solved": f"{exp001['games_solved']}/5", "exp002_solved": f"{exp002['games_solved']}/5"}
Path("/kaggle/working/paper_results_summary.json").write_text(json.dumps(summary, indent=2))
print("Saved: /kaggle/working/paper_results_summary.json")
