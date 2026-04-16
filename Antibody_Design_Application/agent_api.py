import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Extract the first valid JSON object from model output.
    """
    text = str(text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    for block in fenced:
        try:
            return json.loads(block)
        except Exception:
            continue

    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    raise ValueError("LLM response did not contain a valid JSON object.")


def graft_cdrh3_into_heavy(template_heavy: str, template_cdrh3: str, new_cdrh3: str) -> str:
    template_heavy = str(template_heavy).strip().upper()
    template_cdrh3 = str(template_cdrh3).strip().upper()
    new_cdrh3 = str(new_cdrh3).strip().upper()

    if template_cdrh3 not in template_heavy:
        raise ValueError("Template CDRH3 was not found inside the template heavy-chain sequence.")

    return template_heavy.replace(template_cdrh3, new_cdrh3, 1)


def detect_generated_cdr3_column(df: pd.DataFrame) -> str:
    for col in ["cdrh3", "cdr3", "sequence", "generated_cdrh3"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find a generated CDRH3 column in generation output.")


@dataclass
class AgentPlan:
    target_count: int = 100
    min_binding_probability: float = 0.80
    require_hard_filter_pass: bool = True
    require_overall_claim: bool = False
    sampling_mode: str = "sample"
    temperature: float = 1.0
    min_len: int = 8
    num_samples_per_round: int = 128
    max_rounds: int = 5
    deduplicate: bool = True
    sort_binding_weight: float = 1.0
    sort_risk_weight: float = 0.15
    sort_novelty_weight: float = 0.05


class AntibodyDesignAgent:
    """
    Goal-oriented antibody design agent.

    The LLM is only used as:
    - request interpreter
    - plan generator
    - round-by-round controller
    - final summarizer

    The scientific tools remain:
    - generator
    - binder
    - ranker
    """

    def __init__(
        self,
        generator,
        binder,
        ranker,
        llm_model: str = "qwen2.5:1.5b-instruct",
        base_url: str = "http://127.0.0.1:11434/v1",
        api_key: str = "ollama",
        output_dir: str = "outputs",
    ):
        self.generator = generator
        self.binder = binder
        self.ranker = ranker
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.llm_model = llm_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # LLM wrappers
    # =========================
    def _chat_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content or ""
        return _extract_first_json_object(text)

    def _chat_text(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()

    # =========================
    # Planning
    # =========================
    def make_initial_plan(
        self,
        user_request: str,
        antigen_name: str,
        antigen_sequence: str,
        heavy_template: str,
        cdrh3_template: str,
        target_count: int,
        max_rounds: int,
    ) -> AgentPlan:
        system_prompt = """
You are an antibody design agent controller.
Output ONLY one JSON object.
Do not use markdown, bullets, comments, or code fences.

Goal:
Convert the user's design request into practical search parameters.

JSON schema:
{
  "target_count": int,
  "min_binding_probability": float,
  "require_hard_filter_pass": bool,
  "require_overall_claim": bool,
  "sampling_mode": "sample" or "argmax",
  "temperature": float,
  "min_len": int,
  "num_samples_per_round": int,
  "max_rounds": int,
  "deduplicate": bool,
  "sort_binding_weight": float,
  "sort_risk_weight": float,
  "sort_novelty_weight": float
}

Guidance:
- Prefer require_hard_filter_pass = true.
- Prefer require_overall_claim = false unless the request is very strict.
- Prefer sample mode unless the request strongly implies conservative generation.
- Keep temperature within 0.7 to 1.2.
- Keep num_samples_per_round practical.
- sort_binding_weight should dominate.
"""
        user_prompt = f"""
User request:
{user_request}

Context:
- antigen_name: {antigen_name}
- antigen_sequence_length: {len(str(antigen_sequence).strip())}
- heavy_template_length: {len(str(heavy_template).strip())}
- cdrh3_template_length: {len(str(cdrh3_template).strip())}
- fallback_target_count: {target_count}
- fallback_max_rounds: {max_rounds}
"""

        try:
            raw = self._chat_json(system_prompt, user_prompt)
            plan = AgentPlan(
                target_count=int(raw.get("target_count", target_count)),
                min_binding_probability=float(raw.get("min_binding_probability", 0.80)),
                require_hard_filter_pass=bool(raw.get("require_hard_filter_pass", True)),
                require_overall_claim=bool(raw.get("require_overall_claim", False)),
                sampling_mode=str(raw.get("sampling_mode", "sample")),
                temperature=float(raw.get("temperature", 1.0)),
                min_len=int(raw.get("min_len", 8)),
                num_samples_per_round=int(raw.get("num_samples_per_round", 128)),
                max_rounds=int(raw.get("max_rounds", max_rounds)),
                deduplicate=bool(raw.get("deduplicate", True)),
                sort_binding_weight=float(raw.get("sort_binding_weight", 1.0)),
                sort_risk_weight=float(raw.get("sort_risk_weight", 0.15)),
                sort_novelty_weight=float(raw.get("sort_novelty_weight", 0.05)),
            )
            return self._sanitize_plan(plan, fallback_target_count=target_count, fallback_max_rounds=max_rounds)
        except Exception:
            return AgentPlan(
                target_count=target_count,
                min_binding_probability=0.80,
                require_hard_filter_pass=True,
                require_overall_claim=False,
                sampling_mode="sample",
                temperature=1.0,
                min_len=8,
                num_samples_per_round=128,
                max_rounds=max_rounds,
                deduplicate=True,
                sort_binding_weight=1.0,
                sort_risk_weight=0.15,
                sort_novelty_weight=0.05,
            )

    def decide_next_round(
        self,
        current_plan: AgentPlan,
        round_index: int,
        accepted_count: int,
        latest_round_df: pd.DataFrame,
        target_count: int,
    ) -> Dict[str, Any]:
        failure_counts = self._top_failure_reasons(latest_round_df)

        stats = {
            "round_index": int(round_index),
            "accepted_count": int(accepted_count),
            "target_count": int(target_count),
            "remaining_needed": int(max(0, target_count - accepted_count)),
            "generated_this_round": int(len(latest_round_df)),
            "mean_binding_probability": self._safe_mean(latest_round_df, "binding_probability"),
            "mean_risk_score": self._safe_mean(latest_round_df, "developability_risk_score"),
            "hard_filter_pass_rate": self._safe_bool_rate(latest_round_df, "hard_filter_pass"),
            "overall_claim_rate": self._safe_bool_rate(latest_round_df, "overall_claim"),
            "failure_counts": failure_counts,
            "current_plan": asdict(current_plan),
        }

        system_prompt = """
You are controlling an iterative antibody design search.
Output ONLY one JSON object.
No markdown.

JSON schema:
{
  "action": "continue" or "stop",
  "sampling_mode": "sample" or "argmax",
  "temperature": float,
  "num_samples_per_round": int,
  "min_len": int,
  "reason": str
}

Decision policy:
- Stop if enough candidates have been accepted.
- If most failures are due to low binding, reduce temperature slightly and/or increase samples.
- If most failures are due to hard-filter failures, prefer more conservative generation.
- If duplicates dominate, slightly increase exploration.
- Keep values practical.
"""
        user_prompt = json.dumps(stats, indent=2)

        try:
            raw = self._chat_json(system_prompt, user_prompt)
            action = str(raw.get("action", "continue")).lower()
            if accepted_count >= target_count:
                action = "stop"

            return {
                "action": action,
                "sampling_mode": str(raw.get("sampling_mode", current_plan.sampling_mode)),
                "temperature": float(raw.get("temperature", current_plan.temperature)),
                "num_samples_per_round": int(raw.get("num_samples_per_round", current_plan.num_samples_per_round)),
                "min_len": int(raw.get("min_len", current_plan.min_len)),
                "reason": str(raw.get("reason", "LLM controller decision")),
            }
        except Exception:
            if accepted_count >= target_count:
                return {
                    "action": "stop",
                    "sampling_mode": current_plan.sampling_mode,
                    "temperature": current_plan.temperature,
                    "num_samples_per_round": current_plan.num_samples_per_round,
                    "min_len": current_plan.min_len,
                    "reason": "Target count reached.",
                }

            joined = " ".join(failure_counts.keys()).lower()
            new_temp = current_plan.temperature
            new_n = current_plan.num_samples_per_round

            if "binding_too_low" in joined:
                new_temp = max(0.7, current_plan.temperature - 0.1)
                new_n = min(512, int(current_plan.num_samples_per_round * 1.25))
            elif "hard_filter_failed" in joined:
                new_temp = max(0.75, current_plan.temperature - 0.05)
                new_n = min(512, int(current_plan.num_samples_per_round * 1.15))
            elif "duplicate" in joined:
                new_temp = min(1.2, current_plan.temperature + 0.1)
                new_n = min(512, int(current_plan.num_samples_per_round * 1.10))
            else:
                new_n = min(512, int(current_plan.num_samples_per_round * 1.20))

            return {
                "action": "continue",
                "sampling_mode": current_plan.sampling_mode,
                "temperature": new_temp,
                "num_samples_per_round": new_n,
                "min_len": current_plan.min_len,
                "reason": "Fallback heuristic controller.",
            }

    # =========================
    # Core execution
    # =========================
    def run(
        self,
        user_request: str,
        antigen_name: str,
        antigen_sequence: str,
        heavy_template: str,
        cdrh3_template: str,
        target_count: int = 100,
        max_rounds: int = 5,
    ) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
        plan = self.make_initial_plan(
            user_request=user_request,
            antigen_name=antigen_name,
            antigen_sequence=antigen_sequence,
            heavy_template=heavy_template,
            cdrh3_template=cdrh3_template,
            target_count=target_count,
            max_rounds=max_rounds,
        )

        accepted_rows: List[Dict[str, Any]] = []
        history_rows: List[Dict[str, Any]] = []

        seen_heavy = set()
        seen_cdr3 = set()

        for round_idx in range(1, plan.max_rounds + 1):
            gen_df = self.generator.generate(
                antigen=antigen_sequence,
                num_samples=int(plan.num_samples_per_round),
                min_len=int(plan.min_len),
                sample_mode=plan.sampling_mode,
                temperature=float(plan.temperature),
                deduplicate=bool(plan.deduplicate),
            )

            if gen_df is None or len(gen_df) == 0:
                continue

            cdr3_col = detect_generated_cdr3_column(gen_df)

            binding_records = []
            dev_candidates = []

            for i, row in gen_df.iterrows():
                cdr3 = str(row[cdr3_col]).strip().upper()

                if cdr3 in seen_cdr3:
                    continue

                seen_cdr3.add(cdr3)
                candidate_name = f"R{round_idx}_C{i+1}"

                heavy = graft_cdrh3_into_heavy(
                    template_heavy=heavy_template,
                    template_cdrh3=cdrh3_template,
                    new_cdrh3=cdr3,
                )

                bind_result = self.binder.predict(
                    heavy_seq=heavy,
                    antigen_seq=antigen_sequence,
                )

                binding_records.append({
                    "candidate_name": candidate_name,
                    "round": round_idx,
                    "generated_cdrh3": cdr3,
                    "heavy_chain": heavy,
                    "binding_probability": bind_result.get("binding_probability", None),
                    "binding_logit": bind_result.get("logit", None),
                })

                dev_candidates.append({
                    "candidate_name": candidate_name,
                    "Target": antigen_name,
                    "Heavy": heavy,
                    "cdr3": cdr3,
                })

            if len(binding_records) == 0:
                continue

            bind_df = pd.DataFrame(binding_records)

            dev_df = self.ranker.score_candidates(
                target_name=antigen_name,
                candidates=dev_candidates,
            )

            merged = bind_df.merge(dev_df, on="candidate_name", how="left")

            merged["binding_probability"] = pd.to_numeric(
                merged.get("binding_probability", 0.0), errors="coerce"
            ).fillna(0.0)

            merged["developability_risk_score"] = pd.to_numeric(
                merged.get("developability_risk_score", 999.0), errors="coerce"
            ).fillna(999.0)

            merged["cdr3_nn_edit_distance"] = pd.to_numeric(
                merged.get("cdr3_nn_edit_distance", 0.0), errors="coerce"
            ).fillna(0.0)

            merged["hard_filter_pass"] = merged.get("hard_filter_pass", False).astype(bool)
            if "overall_claim" in merged.columns:
                merged["overall_claim"] = merged["overall_claim"].astype(bool)
            else:
                merged["overall_claim"] = False

            merged["composite_score"] = (
                plan.sort_binding_weight * merged["binding_probability"]
                - plan.sort_risk_weight * merged["developability_risk_score"]
                + plan.sort_novelty_weight * merged["cdr3_nn_edit_distance"]
            )

            merged["accepted"] = merged.apply(
                lambda x: self._passes_constraints(x, plan),
                axis=1,
            )

            merged["reject_reason"] = merged.apply(
                lambda x: self._reject_reason(x, plan),
                axis=1,
            )

            merged = merged.sort_values(
                by=["accepted", "composite_score", "binding_probability"],
                ascending=[False, False, False],
            ).reset_index(drop=True)

            history_rows.extend(merged.to_dict(orient="records"))

            for _, row in merged.iterrows():
                if not bool(row["accepted"]):
                    continue

                heavy = str(row["heavy_chain"])
                if heavy in seen_heavy:
                    continue

                seen_heavy.add(heavy)
                accepted_rows.append(row.to_dict())

                if len(accepted_rows) >= plan.target_count:
                    break

            if len(accepted_rows) >= plan.target_count:
                break

            decision = self.decide_next_round(
                current_plan=plan,
                round_index=round_idx,
                accepted_count=len(accepted_rows),
                latest_round_df=merged,
                target_count=plan.target_count,
            )

            if decision["action"] == "stop":
                break

            plan.sampling_mode = "argmax" if str(decision["sampling_mode"]).lower() == "argmax" else "sample"
            plan.temperature = max(0.2, min(2.0, float(decision["temperature"])))
            plan.num_samples_per_round = max(1, min(512, int(decision["num_samples_per_round"])))
            plan.min_len = max(4, min(20, int(decision["min_len"])))

        accepted_df = pd.DataFrame(accepted_rows).reset_index(drop=True)
        history_df = pd.DataFrame(history_rows).reset_index(drop=True)

        if len(accepted_df) > 0:
            accepted_df = accepted_df.sort_values(
                by=["composite_score", "binding_probability"],
                ascending=[False, False],
            ).reset_index(drop=True)

        accepted_csv = self.output_dir / "agent_accepted_candidates.csv"
        history_csv = self.output_dir / "agent_full_history.csv"

        accepted_df.to_csv(accepted_csv, index=False)
        history_df.to_csv(history_csv, index=False)

        summary = self._final_summary_text(
            user_request=user_request,
            antigen_name=antigen_name,
            accepted_df=accepted_df,
            history_df=history_df,
            accepted_csv=str(accepted_csv),
            history_csv=str(history_csv),
        )

        return summary, accepted_df, history_df

    # =========================
    # Rules
    # =========================
    def _passes_constraints(self, row: pd.Series, plan: AgentPlan) -> bool:
        if float(row.get("binding_probability", 0.0)) < float(plan.min_binding_probability):
            return False

        if plan.require_hard_filter_pass and not bool(row.get("hard_filter_pass", False)):
            return False

        if plan.require_overall_claim and not bool(row.get("overall_claim", False)):
            return False

        return True

    def _reject_reason(self, row: pd.Series, plan: AgentPlan) -> str:
        if float(row.get("binding_probability", 0.0)) < float(plan.min_binding_probability):
            return "binding_too_low"

        if plan.require_hard_filter_pass and not bool(row.get("hard_filter_pass", False)):
            return "hard_filter_failed"

        if plan.require_overall_claim and not bool(row.get("overall_claim", False)):
            return "overall_claim_failed"

        return ""

    # =========================
    # Utils
    # =========================
    def _sanitize_plan(self, plan: AgentPlan, fallback_target_count: int, fallback_max_rounds: int) -> AgentPlan:
        plan.target_count = max(1, min(1000, int(plan.target_count or fallback_target_count)))
        plan.min_binding_probability = max(0.0, min(1.0, float(plan.min_binding_probability)))
        plan.sampling_mode = "argmax" if str(plan.sampling_mode).lower() == "argmax" else "sample"
        plan.temperature = max(0.2, min(2.0, float(plan.temperature)))
        plan.min_len = max(4, min(20, int(plan.min_len)))
        plan.num_samples_per_round = max(1, min(512, int(plan.num_samples_per_round)))
        plan.max_rounds = max(1, min(20, int(plan.max_rounds or fallback_max_rounds)))
        plan.sort_binding_weight = float(plan.sort_binding_weight)
        plan.sort_risk_weight = float(plan.sort_risk_weight)
        plan.sort_novelty_weight = float(plan.sort_novelty_weight)
        return plan

    def _safe_mean(self, df: pd.DataFrame, col: str) -> Optional[float]:
        if col not in df.columns or len(df) == 0:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            return None
        return float(s.mean())

    def _safe_bool_rate(self, df: pd.DataFrame, col: str) -> Optional[float]:
        if col not in df.columns or len(df) == 0:
            return None
        s = df[col].astype(bool)
        return float(s.mean())

    def _top_failure_reasons(self, df: pd.DataFrame) -> Dict[str, int]:
        if "reject_reason" not in df.columns or len(df) == 0:
            return {}
        s = df["reject_reason"].fillna("").astype(str)
        s = s[s != ""]
        if len(s) == 0:
            return {}
        vc = s.value_counts().head(5)
        return {str(k): int(v) for k, v in vc.items()}

    def _final_summary_text(
        self,
        user_request: str,
        antigen_name: str,
        accepted_df: pd.DataFrame,
        history_df: pd.DataFrame,
        accepted_csv: str,
        history_csv: str,
    ) -> str:
        system_prompt = """
You are writing a concise execution summary for an antibody design agent.
Be short, factual, and practical.
"""
        user_prompt = f"""
User request:
{user_request}

Antigen name:
{antigen_name}

Accepted count:
{len(accepted_df)}

Total evaluated:
{len(history_df)}

Top accepted candidates (first 5 as JSON):
{accepted_df.head(5).to_json(orient="records") if len(accepted_df) > 0 else "[]"}

Accepted CSV:
{accepted_csv}

History CSV:
{history_csv}

Write a short summary that states:
- whether the target count was met
- how many candidates were accepted
- how many total were evaluated
- where outputs were saved
"""
        try:
            return self._chat_text(system_prompt, user_prompt)
        except Exception:
            target_met = "yes" if len(accepted_df) > 0 else "no"
            return (
                f"Agent completed.\n"
                f"Target met: {target_met}\n"
                f"Accepted candidates: {len(accepted_df)}\n"
                f"Total evaluated: {len(history_df)}\n"
                f"Accepted CSV: {accepted_csv}\n"
                f"History CSV: {history_csv}"
            )