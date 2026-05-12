"""llm_backend.py -- local quantized LLM inference for the ARC-AGI-3 agent.

Provides a single LLMBackend class wrapping either llama.cpp (via llama-cpp-python)
or vLLM. Both expose the same .generate() interface so agent.py is decoupled from
the inference engine.

Target models (GGUF Q4_K_M or similar):
    - Qwen2.5-7B-Instruct
    - Llama-3.1-8B-Instruct

Kaggle constraint: no internet at eval time. All weights must be bundled as a
Kaggle dataset and loaded from a local path.

Optional prompt/response log (JSONL, one record per line):
    {"ts": <float>, "prompt_chars": <int>, "response": <str>, "elapsed_ms": <float>}
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Backend selector constants
# ---------------------------------------------------------------------------

BACKEND_LLAMACPP = "llamacpp"
BACKEND_VLLM = "vllm"
BACKEND_AIRLLM = "airllm"
BACKEND_STUB = "stub"        # no model loaded; returns scripted responses for scaffold testing
BACKEND_ANTHROPIC = "anthropic"  # calls Claude API — local dev only, swap for llamacpp on Kaggle
BACKEND_TRANSFORMERS = "transformers"  # HuggingFace transformers — runs on Kaggle T4 GPU


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LLMBackend:
    """Local quantized LLM inference wrapper.

    Parameters
    ----------
    model_path:
        Path to GGUF model file (llama.cpp) or HuggingFace model dir (vLLM).
    backend:
        "llamacpp" (default, CPU+GPU) or "vllm" (GPU-only, higher throughput).
    n_ctx:
        Context window in tokens.
    n_gpu_layers:
        Layers to offload to GPU. -1 = all, 0 = CPU-only.
    max_tokens:
        Max tokens to generate per call.
    temperature:
        Sampling temperature. Lower = more deterministic.
    log_path:
        If set, appends each prompt/response pair to this JSONL file.
    """

    def __init__(
        self,
        model_path: str | Path,
        backend: str = BACKEND_LLAMACPP,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        max_tokens: int = 512,
        temperature: float = 0.2,
        log_path: Path | str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.backend = backend
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.log_path = Path(log_path) if log_path else None

        self._model: Any = None  # lazy-loaded on first generate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Run inference and return generated text (stripped).

        Parameters
        ----------
        prompt:
            Full prompt string (system + history + current state formatted by caller).
        max_tokens:
            Per-call override of instance default.
        """
        if self._model is None:
            self._load_model()

        n_tok = max_tokens or self.max_tokens
        t0 = time.perf_counter()

        if self.backend == BACKEND_LLAMACPP:
            response = self._generate_llamacpp(prompt, n_tok)
        elif self.backend == BACKEND_VLLM:
            response = self._generate_vllm(prompt, n_tok)
        elif self.backend == BACKEND_AIRLLM:
            response = self._generate_airllm(prompt, n_tok)
        elif self.backend == BACKEND_STUB:
            response = self._generate_stub(prompt, n_tok)
        elif self.backend == BACKEND_ANTHROPIC:
            response = self._generate_anthropic(prompt, n_tok)
        elif self.backend == BACKEND_TRANSFORMERS:
            response = self._generate_transformers(prompt, n_tok)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if self.log_path is not None:
            self._log(prompt, response, elapsed_ms)

        return response.strip()

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        """Release the model from memory (free VRAM between games if needed)."""
        self._model = None

    # ------------------------------------------------------------------
    # Engine-specific generation
    # ------------------------------------------------------------------

    def _generate_llamacpp(self, prompt: str, max_tokens: int) -> str:
        output = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=self.temperature,
            stop=["<|eot_id|>", "<|im_end|>", "```\n\n"],
            echo=False,
        )
        return output["choices"][0]["text"]

    def _generate_vllm(self, prompt: str, max_tokens: int) -> str:
        from vllm import SamplingParams  # type: ignore[import]
        params = SamplingParams(max_tokens=max_tokens, temperature=self.temperature)
        outputs = self._model.generate([prompt], params)
        return outputs[0].outputs[0].text

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self.backend == BACKEND_STUB:
            self._load_stub()
            return
        if self.backend == BACKEND_ANTHROPIC:
            self._load_anthropic()
            return
        if self.backend == BACKEND_TRANSFORMERS:
            self._load_transformers()
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Download a GGUF file or point model_path at a HuggingFace directory."
            )
        if self.backend == BACKEND_LLAMACPP:
            self._load_llamacpp()
        elif self.backend == BACKEND_VLLM:
            self._load_vllm()
        elif self.backend == BACKEND_AIRLLM:
            self._load_airllm()
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    def _load_llamacpp(self) -> None:
        try:
            from llama_cpp import Llama  # pip install llama-cpp-python
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            ) from exc
        self._model = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

    def _load_vllm(self) -> None:
        try:
            from vllm import LLM  # pip install vllm
        except ImportError as exc:
            raise ImportError(
                "vllm not installed. Run: pip install vllm"
            ) from exc
        self._model = LLM(model=str(self.model_path), max_model_len=self.n_ctx)

    def _load_anthropic(self) -> None:
        try:
            import anthropic  # pip install anthropic
        except ImportError as exc:
            raise ImportError("anthropic not installed. Run: pip install anthropic") from exc
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set in environment")
        self._model = anthropic.Anthropic(api_key=api_key)
        # model_path is repurposed as model name for the Anthropic backend
        self._anthropic_model = str(self.model_path) if self.model_path.name != "." else "claude-haiku-4-5-20251001"

    def _generate_anthropic(self, prompt: str, max_tokens: int) -> str:
        msg = self._model.messages.create(
            model=self._anthropic_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _load_stub(self) -> None:
        # Per-game state: which actions advanced levels, cycle index
        self._model = {"good_actions": {}, "cycle_idx": {}}

    def _generate_stub(self, prompt: str, max_tokens: int) -> str:
        """Systematic explorer: cycles actions, exploits ones that advance levels.

        No LLM needed. Detects level progression from trajectory in prompt,
        builds per-game good-action list, cycles through it preferentially.
        """
        import re

        # Parse available_actions from prompt
        avail = ["ACTION1"]
        m = re.search(r'"available_actions":\s*\[([^\]]+)\]', prompt)
        if m:
            vals = [v.strip() for v in m.group(1).split(",") if v.strip()]
            try:
                from arcengine.enums import GameAction
                v2n = {str(a.value): a.name for a in GameAction}
                avail = [v2n[v] for v in vals if v in v2n] or ["ACTION1"]
            except Exception:
                pass

        # Parse game name
        gm = re.search(r'game called "([^"]+)"', prompt)
        game = gm.group(1) if gm else "default"

        state = self._model
        if game not in state["good_actions"]:
            state["good_actions"][game] = []
            state["cycle_idx"][game] = 0

        good = state["good_actions"][game]

        # Detect level advance from trajectory
        levels = re.findall(r'"levels_completed":\s*(\d+)', prompt)
        if len(levels) >= 2 and int(levels[-1]) > int(levels[-2]):
            acts = re.findall(r'step \d+: (ACTION\d+)', prompt)
            if acts and acts[-1] not in good:
                good.append(acts[-1])

        # Exploit good actions first; otherwise cycle through available
        pool = good if good else avail
        action = pool[state["cycle_idx"][game] % len(pool)]
        state["cycle_idx"][game] += 1

        action_str = f"{action} x=32 y=32" if action == "ACTION6" else action

        if "budget remaining" in prompt or "Complete the current level" in prompt:
            items = [f"{a} x=32 y=32" if a == "ACTION6" else a for a in avail * 4]
            return f"PLAN: {','.join(items)}\nCONFIDENCE: LOW\nFALLBACK: ACTION1"

        return (
            f"HYPOTHESIS: Cycling actions; good so far: {good or 'none'}\n"
            f"UNCERTAIN: Which actions advance each level\n"
            f"NEXT_ACTION: {action_str}\n"
            f"REASON: Systematic exploration"
        )

    def _load_airllm(self) -> None:
        try:
            import airllm
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "airllm not installed. Run: pip install airllm sentencepiece"
            ) from exc
        mp = str(self.model_path)
        if mp.endswith(".gguf"):
            raise ValueError(
                "airllm requires HuggingFace model format, not .gguf. "
                "Use --backend llamacpp for GGUF files."
            )
        self._tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
        self._model = airllm.AutoModel.from_pretrained(mp)

    def _generate_airllm(self, prompt: str, max_tokens: int) -> str:
        import torch
        input_ids = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.n_ctx,
        ).input_ids
        with torch.no_grad():
            out = self._model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                use_cache=False,               # required — airllm manages its own cache
                return_dict_in_generate=False,
            )
        new_tokens = out[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _load_transformers(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as exc:
            raise ImportError("transformers not installed. Run: pip install transformers torch") from exc
        model_name = str(self.model_path)  # e.g. "Qwen/Qwen2.5-7B-Instruct" or "/kaggle/input/..."
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # 4-bit quantization for memory-constrained GPUs (Kaggle T4 = 15.6 GB VRAM).
        # Qwen2.5-7B at BF16 ≈ 14 GB; 4-bit ≈ 4.5 GB — use when load_in_4bit=True.
        if getattr(self, "load_in_4bit", False):
            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_cfg,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except ImportError:
                # bitsandbytes not installed — fall back to float16
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float16,
                    device_map="auto", trust_remote_code=True,
                )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )

    def _generate_transformers(self, prompt: str, max_tokens: int) -> str:
        import torch
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
        new_tokens = out[0][inputs.input_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, prompt: str, response: str, elapsed_ms: float) -> None:
        assert self.log_path is not None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "prompt_chars": len(prompt),
            "response": response,
            "elapsed_ms": round(elapsed_ms, 1),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Prompt builders (called by agent.py)
# ---------------------------------------------------------------------------

def format_hypothesis_prompt(
    game_name: str,
    trajectory: list[dict],
    current_hypothesis: str,
    current_obs: dict,
    grid_text: str = "",
) -> str:
    """Prompt for the hypothesis-update step after each action.

    Parameters
    ----------
    game_name:
        Environment name, e.g. "ls20".
    trajectory:
        List of dicts from ArcEnvWrapper.trajectory_summary().
    current_hypothesis:
        Agent's current theory of the environment rules (plain text).
    current_obs:
        Latest observation dict.
    grid_text:
        32x32 ARC color grid as compact text (from ArcEnvWrapper.grid_as_text).
    """
    traj_text = "\n".join(
        f"  step {s['step']}: {s['action']} -> {s['obs']}"
        for s in trajectory
    )
    return f"""You are an agent exploring a novel turn-based game called "{game_name}".
You have no prior knowledge of its rules. Infer everything from observations.

## Action space
ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION7 (discrete)
ACTION6 (requires x=<int> y=<int>)

## Trajectory so far
{traj_text or "(no steps yet)"}

## Current observation
{json.dumps(current_obs, default=str, indent=2)}

## Current game grid (ARC colors 0-9: 0=black 1=blue 2=red 3=green 4=yellow 5=grey 6=magenta 7=orange 8=azure 9=maroon)
{grid_text or "(not available)"}

## Current hypothesis
{current_hypothesis or "(none yet)"}

Update the hypothesis and propose the next action. Respond exactly as:
HYPOTHESIS: <updated theory of rules>
UNCERTAIN: <what you still don't know>
NEXT_ACTION: <ACTION1|ACTION2|ACTION3|ACTION4|ACTION5|ACTION6 x=N y=N|ACTION7>
REASON: <why this action best resolves uncertainty or advances toward the goal>
"""


def format_plan_prompt(
    game_name: str,
    hypothesis: str,
    current_obs: dict,
    budget_remaining: int,
) -> str:
    """Prompt for the planning step once the world model is stable.

    Called when the agent has enough confidence to plan toward the goal
    rather than continuing exploration.
    """
    return f"""You are an agent that has inferred the rules of a game called "{game_name}".
Complete the current level in as few actions as possible.

## Inferred rules (world model)
{hypothesis}

## Current observation
{json.dumps(current_obs, default=str, indent=2)}

## Action budget remaining
{budget_remaining} actions

Output a minimal action sequence. Efficiency is scored against human median.
Respond exactly as:
PLAN: <comma-separated actions, e.g. ACTION1,ACTION3,ACTION6 x=2 y=4,ACTION2>
CONFIDENCE: <HIGH|MEDIUM|LOW>
FALLBACK: <what to try if the plan fails on step 1>
"""
