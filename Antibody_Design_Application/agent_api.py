import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI


def _extract_first_json_object(text: str) -> Dict[str, Any]:
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


def extract_target_count_from_request(user_request: str) -> Optional[int]:
    text = str(user_request or "").strip().lower()
    if not text:
        return None

    patterns = [
        r"\bfind\s+(\d{1,4})\s+(?:antibody\s+)?candidates?\b",
        r"\bfind\s+(\d{1,4})\s+(?:antibodies|antibody)\b",
        r"\bgenerate\s+(\d{1,4})\s+(?:antibody\s+)?candidates?\b",
        r"\bgenerate\s+(\d{1,4})\s+(?:antibodies|antibody)\b",
        r"\bdesign\s+(\d{1,4})\s+(?:antibody\s+)?candidates?\b",
        r"\bdesign\s+(\d{1,4})\s+(?:antibodies|antibody)\b",
        r"\bidentify\s+(\d{1,4})\s+(?:antibody\s+)?candidates?\b",
        r"\b(\d{1,4})\s+(?:antibody\s+)?candidates?\b",
        r"\b(\d{1,4})\s+(?:antibodies|antibody)\b",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            try:
                value = int(m.group(1))
                if 1 <= value <= 1000:
                    return value
            except Exception:
                continue

    return None


@dataclass
class AgentPlan:
    target_count: int = 10
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

    def _chat_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.25) -> str:
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    def make_initial_plan(
        self,
        user_request: str,
        antigen_name: str,
        antigen_sequence: str,
        heavy_template: str,
        cdrh3_template: str,
        default_target_count: int,
        max_rounds: int,
    ) -> AgentPlan:
        parsed_target_count = extract_target_count_from_request(user_request)

        system_prompt = """
You are an antibody design agent controller.
Output ONLY one JSON object.

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
"""
        user_prompt = f"""
User request:
{user_request}

Context:
- antigen_name: {antigen_name}
- antigen_sequence_length: {len(str(antigen_sequence).strip())}
- heavy_template_length: {len(str(heavy_template).strip())}
- cdrh3_template_length: {len(str(cdrh3_template).strip())}
- parsed_target_count_from_request: {parsed_target_count}
- fallback_target_count: {default_target_count}
- fallback_max_rounds: {max_rounds}
"""

        fallback_target = parsed_target_count if parsed_target_count is not None else default_target_count

        try:
            raw = self._chat_json(system_prompt, user_prompt)
            plan = AgentPlan(
                target_count=int(raw.get("target_count", fallback_target)),
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
            if parsed_target_count is not None:
                plan.target_count = parsed_target_count
            return self._sanitize_plan(plan, fallback_target, max_rounds)
        except Exception:
            return AgentPlan(
                target_count=fallback_target,
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

JSON schema:
{
  "action": "continue" or "stop",
  "sampling_mode": "sample" or "argmax",
  "temperature": float,
  "num_samples_per_round": int,
  "min_len": int,
  "reason": str
}
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

            return {
                "action": "continue",
                "sampling_mode": current_plan.sampling_mode,
                "temperature": current_plan.temperature,
                "num_samples_per_round": min(512, int(current_plan.num_samples_per_round * 1.2)),
                "min_len": current_plan.min_len,
                "reason": "Fallback heuristic controller.",
            }

    def run(
        self,
        user_request: str,
        antigen_name: str,
        antigen_sequence: str,
        heavy_template: str,
        cdrh3_template: str,
        default_target_count: int = 10,
        max_rounds: int = 5,
    ) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
        plan = self.make_initial_plan(
            user_request=user_request,
            antigen_name=antigen_name,
            antigen_sequence=antigen_sequence,
            heavy_template=heavy_template,
            cdrh3_template=cdrh3_template,
            default_target_count=default_target_count,
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
            dev_df = self.ranker.score_candidates(target_name=antigen_name, candidates=dev_candidates)
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

            merged["accepted"] = merged.apply(lambda x: self._passes_constraints(x, plan), axis=1)
            merged["reject_reason"] = merged.apply(lambda x: self._reject_reason(x, plan), axis=1)

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
            plan=plan,
            accepted_df=accepted_df,
            history_df=history_df,
            accepted_csv=str(accepted_csv),
            history_csv=str(history_csv),
        )

        return summary, accepted_df, history_df

    def run_analysis(self, analysis_type: str, accepted_df: pd.DataFrame, history_df: pd.DataFrame) -> str:
        accepted_df = accepted_df if accepted_df is not None else pd.DataFrame()
        history_df = history_df if history_df is not None else pd.DataFrame()

        mapping = {
            "summary": self._analysis_summary,
            "ranking": self._analysis_ranking,
            "bottleneck": self._analysis_bottleneck,
            "threshold": self._analysis_threshold,
            "sampling_strategy": self._analysis_sampling_strategy,
            "acceptance_diagnostics": self._analysis_acceptance_diagnostics,
            "round_trend": self._analysis_round_trend,
        }

        fn = mapping.get(analysis_type)
        if fn is None:
            return f"Unknown analysis type: {analysis_type}"

        return fn(accepted_df, history_df)
    
    def answer_question(
        self,
        question: str,
        user_request: str = "",
        antigen_name: str = "",
        latest_summary: str = "",
        accepted_df: Optional[pd.DataFrame] = None,
        history_df: Optional[pd.DataFrame] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        accepted_df = accepted_df if accepted_df is not None else pd.DataFrame()
        history_df = history_df if history_df is not None else pd.DataFrame()
        chat_history = chat_history or []

        accepted_preview = accepted_df.head(8).to_dict(orient="records") if len(accepted_df) > 0 else []
        history_preview = history_df.head(12).to_dict(orient="records") if len(history_df) > 0 else []
        reject_counts = self._reject_counts(history_df)
        accepted_stats = self._build_table_stats(accepted_df)
        history_stats = self._build_table_stats(history_df)
        history_text = self._chat_history_to_text(chat_history, max_turns=6)

        system_prompt = """
You are an antibody design analysis assistant.

Your role:
- answer open-ended questions about the current antibody-design run,
- explain accepted candidates, rejected candidates, bottlenecks, thresholds, and round trends,
- stay grounded in the provided run results,
- do not invent facts not supported by the provided tables and summary,
- do not switch to unrelated generic AI/ML explanations.

Rules:
- If there are no accepted candidates, say so clearly.
- Prefer concise, factual answers.
- If the user asks for interpretation, base it on the provided statistics.
- If evidence is insufficient, explicitly say what is known and what is uncertain.
"""

        user_prompt = f"""
User question:
{question}

Current design request:
{user_request}

Current antigen:
{antigen_name}

Latest run summary:
{latest_summary}

Accepted stats:
{json.dumps(accepted_stats, ensure_ascii=False, indent=2)}

History stats:
{json.dumps(history_stats, ensure_ascii=False, indent=2)}

Reject counts:
{json.dumps(reject_counts, ensure_ascii=False, indent=2)}

Accepted preview:
{json.dumps(accepted_preview, ensure_ascii=False, indent=2)}

History preview:
{json.dumps(history_preview, ensure_ascii=False, indent=2)}

Recent conversation:
{history_text}

Please answer the user's question based only on the current antibody-design task and results.
"""

        try:
            return self._chat_text(system_prompt, user_prompt, temperature=0.2)
        except Exception as e:
            return f"Error while generating answer: {str(e)}"

    def _chat_history_to_text(self, chat_history: List[Dict[str, str]], max_turns: int = 6) -> str:
        if not chat_history:
            return ""

        trimmed = chat_history[-2 * max_turns:]
        parts = []

        for msg in trimmed:
            if not isinstance(msg, dict):
                continue

            role = str(msg.get("role", "")).strip().lower()
            content = self._normalize_chat_content(msg.get("content", ""))

            if not content:
                continue

            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"{role}: {content}")

        return "\n".join(parts)

    def _normalize_chat_content(self, raw: Any) -> str:
        if isinstance(raw, str):
            return raw.strip()

        if isinstance(raw, list):
            texts = []
            for item in raw:
                if isinstance(item, dict):
                    if "text" in item:
                        texts.append(str(item["text"]))
                    elif "content" in item:
                        texts.append(str(item["content"]))
                else:
                    texts.append(str(item))
            return " ".join([t for t in texts if str(t).strip()]).strip()

        if isinstance(raw, dict):
            if "text" in raw:
                return str(raw.get("text", "")).strip()
            if "content" in raw:
                return str(raw.get("content", "")).strip()
            return json.dumps(raw, ensure_ascii=False)

        return str(raw).strip()
    
    
    
    
    

    def _analysis_summary(self, accepted_df: pd.DataFrame, history_df: pd.DataFrame) -> str:
        accepted_n = len(accepted_df)
        history_n = len(history_df)

        lines = [
            "Run summary:",
            f"- Accepted candidates: {accepted_n}",
            f"- Total evaluated candidates: {history_n}",
        ]

        if accepted_n == 0:
            lines.append("- No accepted candidates are available.")
            return "\n".join(lines)

        top_df = accepted_df.head(2)
        lines.append("- Top accepted candidates:")
        for i, (_, row) in enumerate(top_df.iterrows(), start=1):
            lines.append(self._format_candidate_line(i, row))
        return "\n".join(lines)

    def _analysis_ranking(self, accepted_df: pd.DataFrame, history_df: pd.DataFrame) -> str:
        if len(accepted_df) == 0:
            return "No accepted candidates are available."

        top_df = accepted_df.head(5)
        lines = [f"Top {len(top_df)} accepted candidates:"]
        for i, (_, row) in enumerate(top_df.iterrows(), start=1):
            lines.append(self._format_candidate_line(i, row))
        return "\n".join(lines)

    def _analysis_bottleneck(self, accepted_df: pd.DataFrame, history_df: pd.DataFrame) -> str:
        reject_counts = self._reject_counts(history_df)
        if not reject_counts:
            return "No clear bottleneck could be identified because no rejection statistics are available."

        bottleneck, count = max(reject_counts.items(), key=lambda x: x[1])
        return (
            f"The dominant bottleneck in this run is '{bottleneck}', "
            f"which affected {count} evaluated candidates.\n"
            f"Reject counts: {reject_counts}"
        )

    def _analysis_threshold(self, accepted_df: pd.DataFrame, history_df: pd.DataFrame) -> str:
        if len(history_df) == 0:
            return "I cannot assess threshold strictness because no run history is available."

        reject_counts = self._reject_counts(history_df)
        total = len(history_df)
        binding_low = reject_counts.get("binding_too_low", 0)
        hard_fail = reject_counts.get("hard_filter_failed", 0)

        binding_rate = binding_low / total if total > 0 else 0.0
        hard_rate = hard_fail / total if total > 0 else 0.0

        lines = [
            f"Binding-too-low rejection rate: {binding_rate:.1%}",
            f"Hard-filter-failed rejection rate: {hard_rate:.1%}",
        ]

        if binding_rate >= 0.5:
            lines.append("The binding threshold appears strict for this run.")
        elif binding_rate >= 0.25:
            lines.append("The binding threshold appears moderately restrictive.")
        else:
            lines.append("The binding threshold does not appear to be the main bottleneck.")

        return "\n".join(lines)

    def _analysis_sampling_strategy(self, accepted_df: pd.DataFrame, history_df: pd.DataFrame) -> str:
        if len(history_df) == 0:
            return "I cannot assess sampling strategy because no run history is available."

        acc_rate = float(history_df["accepted"].mean()) if "accepted" in history_df.columns else None
        reject_counts = self._reject_counts(history_df)

        if acc_rate is None:
            return "I cannot assess sampling strategy because acceptance rate is unavailable."

        lines = [f"Acceptance rate: {acc_rate:.1%}"]

        if acc_rate < 0.15:
            lines.append("Increasing sampling is likely reasonable because the acceptance rate is low.")
        else:
            lines.append("Increasing sampling is not the first change I would make.")

        if reject_counts:
            dominant_reason = max(reject_counts.items(), key=lambda x: x[1])[0]
            lines.append(f"Dominant rejection reason: {dominant_reason}")
            if dominant_reason == "hard_filter_failed":
                lines.append("Improving candidate quality may help more than only increasing sample count.")
            elif dominant_reason == "binding_too_low":
                lines.append("More sampling may help, but the binding constraint may also be restrictive.")

        return "\n".join(lines)

    def _analysis_acceptance_diagnostics(self, accepted_df: pd.DataFrame, history_df: pd.DataFrame) -> str:
        if len(history_df) == 0:
            return "No diagnostics are available because no run history exists."

        accepted_stats = self._build_table_stats(accepted_df)
        history_stats = self._build_table_stats(history_df)
        reject_counts = self._reject_counts(history_df)

        return (
            "Acceptance diagnostics:\n"
            f"- Accepted stats: {json.dumps(accepted_stats, ensure_ascii=False)}\n"
            f"- History stats: {json.dumps(history_stats, ensure_ascii=False)}\n"
            f"- Reject counts: {json.dumps(reject_counts, ensure_ascii=False)}"
        )

    def _analysis_round_trend(self, accepted_df: pd.DataFrame, history_df: pd.DataFrame) -> str:
        if len(history_df) == 0 or "round" not in history_df.columns:
            return "Round trend analysis is unavailable because no per-round history was recorded."

        lines = ["Round trend:"]
        grouped = history_df.groupby("round")
        for round_id, df_round in grouped:
            mean_binding = self._safe_mean(df_round, "binding_probability")
            mean_risk = self._safe_mean(df_round, "developability_risk_score")
            accepted_rate = self._safe_bool_rate(df_round, "accepted")
            lines.append(
                f"- Round {int(round_id)}: "
                f"mean_binding_probability={self._format_float(mean_binding)}, "
                f"mean_developability_risk_score={self._format_float(mean_risk)}, "
                f"accepted_rate={self._format_float(accepted_rate, 3)}"
            )
        return "\n".join(lines)

    def explain_request(
        self,
        user_request: str,
        antigen_name: str = "",
    ) -> str:
        parsed_target_count = extract_target_count_from_request(user_request)
        if parsed_target_count is None:
            return (
                f"The request asks the agent to design antibody candidates for {antigen_name}. "
                "No explicit target count was detected, so the default count is 10."
            )
        return (
            f"The request asks the agent to design antibody candidates for {antigen_name}. "
            f"An explicit target count of {parsed_target_count} was detected."
        )

    def _format_candidate_line(self, idx: int, row: pd.Series) -> str:
        candidate_name = row.get("candidate_name", f"C{idx}")
        cdrh3 = row.get("generated_cdrh3", row.get("cdr3", ""))
        binding = row.get("binding_probability", None)
        risk = row.get("developability_risk_score", None)
        hard_filter = row.get("hard_filter_pass", None)
        novelty = row.get("cdr3_nn_edit_distance", None)

        return (
            f"{idx}. {candidate_name}: "
            f"CDRH3={cdrh3}; "
            f"binding_probability={self._format_float(binding)}; "
            f"developability_risk_score={self._format_float(risk)}; "
            f"hard_filter_pass={hard_filter}; "
            f"cdr3_nn_edit_distance={self._format_float(novelty)}"
        )

    def _format_float(self, value: Any, digits: int = 3) -> str:
        try:
            if value is None or pd.isna(value):
                return "NA"
            return f"{float(value):.{digits}f}"
        except Exception:
            return str(value)

    def _reject_counts(self, history_df: pd.DataFrame) -> Dict[str, int]:
        if history_df is None or len(history_df) == 0 or "reject_reason" not in history_df.columns:
            return {}
        s = history_df["reject_reason"].fillna("").astype(str)
        s = s[s != ""]
        if len(s) == 0:
            return {}
        vc = s.value_counts().head(10)
        return {str(k): int(v) for k, v in vc.items()}

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

    def _sanitize_plan(self, plan: AgentPlan, fallback_target_count: int, fallback_max_rounds: int) -> AgentPlan:
        plan.target_count = max(1, min(1000, int(plan.target_count or fallback_target_count)))
        plan.min_binding_probability = max(0.0, min(1.0, float(plan.min_binding_probability)))
        plan.sampling_mode = "argmax" if str(plan.sampling_mode).lower() == "argmax" else "sample"
        plan.temperature = max(0.2, min(2.0, float(plan.temperature)))
        plan.min_len = max(4, min(20, int(plan.min_len)))
        plan.num_samples_per_round = max(1, min(512, int(plan.num_samples_per_round)))
        plan.max_rounds = max(1, min(20, int(plan.max_rounds or fallback_max_rounds)))
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

    def _build_table_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or len(df) == 0:
            return {
                "rows": 0,
                "mean_binding_probability": None,
                "mean_developability_risk_score": None,
                "hard_filter_pass_rate": None,
                "overall_claim_rate": None,
                "accepted_rate": None,
                "round_counts": {},
            }

        round_counts = {}
        if "round" in df.columns:
            vc = df["round"].value_counts().sort_index()
            round_counts = {str(int(k)): int(v) for k, v in vc.items()}

        return {
            "rows": int(len(df)),
            "mean_binding_probability": self._safe_mean(df, "binding_probability"),
            "mean_developability_risk_score": self._safe_mean(df, "developability_risk_score"),
            "hard_filter_pass_rate": self._safe_bool_rate(df, "hard_filter_pass"),
            "overall_claim_rate": self._safe_bool_rate(df, "overall_claim"),
            "accepted_rate": self._safe_bool_rate(df, "accepted"),
            "round_counts": round_counts,
        }

    def _final_summary_text(
        self,
        user_request: str,
        antigen_name: str,
        plan: AgentPlan,
        accepted_df: pd.DataFrame,
        history_df: pd.DataFrame,
        accepted_csv: str,
        history_csv: str,
    ) -> str:
        target_met = "yes" if len(accepted_df) >= int(plan.target_count) else "no"
        return (
            f"Agent completed.\n"
            f"Planned target count: {plan.target_count}\n"
            f"Target met: {target_met}\n"
            f"Accepted candidates: {len(accepted_df)}\n"
            f"Total evaluated: {len(history_df)}\n"
            f"Accepted CSV: {accepted_csv}\n"
            f"History CSV: {history_csv}"
        )
