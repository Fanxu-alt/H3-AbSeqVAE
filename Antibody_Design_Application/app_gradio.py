from pathlib import Path

import gradio as gr
import pandas as pd

from generate_api import AntibodyGenerator
from binder_api import AntibodyBinder
from developability_api import DevelopabilityRanker

GEN_MODEL_PATH = "models/conditional_cvae_finetune.pt"
BINDER_MODEL_PATH = "models/best_esm2_cross_attention.pt"
DEV_CSV_PATH = "filtered_Label_1.csv"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLE_ANTIGEN = (
    "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
)

EXAMPLE_TARGET = "SARS-CoV2_Beta"

EXAMPLE_HEAVY = (
    "EVQLVESGGGLVQPGGSLRLSCAASGITVSSNYMTWVRQAPGKGLEWVSVIYSGGSTFYADSVRGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDLEMAGAFDIWGQGTMVTVSS"
)

EXAMPLE_CDRH3 = "ARDLEMAGAFDI"

generator = AntibodyGenerator(GEN_MODEL_PATH)
binder = AntibodyBinder(BINDER_MODEL_PATH)
ranker = DevelopabilityRanker(DEV_CSV_PATH)

AVAILABLE_TARGETS = ranker.list_targets()
DEFAULT_TARGET = EXAMPLE_TARGET if EXAMPLE_TARGET in AVAILABLE_TARGETS else (
    AVAILABLE_TARGETS[0] if AVAILABLE_TARGETS else None
)

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

def run_generation(antigen, num_samples, min_len, sample_mode, temperature, deduplicate):
    try:
        if not antigen or not str(antigen).strip():
            return "Please provide an antigen sequence.", pd.DataFrame()

        df = generator.generate(
            antigen=antigen,
            num_samples=int(num_samples),
            min_len=int(min_len),
            sample_mode=sample_mode,
            temperature=float(temperature),
            deduplicate=bool(deduplicate),
        )

        out_csv = OUTPUT_DIR / "generated_candidates.csv"
        df.to_csv(out_csv, index=False)

        summary = (
            f"Generated {len(df)} candidate CDRH3 sequences\n"
            f"Sampling mode: {sample_mode}\n"
            f"Temperature: {float(temperature):.2f}\n"
            f"Saved to: {out_csv}"
        )
        return summary, df

    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()


def load_generate_example():
    return EXAMPLE_ANTIGEN, 32, 8, "sample", 1.0, True

def run_binding_prediction(heavy, antigen):
    try:
        if not heavy or not str(heavy).strip():
            return "Please provide a heavy-chain sequence.", pd.DataFrame()
        if not antigen or not str(antigen).strip():
            return "Please provide an antigen sequence.", pd.DataFrame()

        result = binder.predict(heavy_seq=heavy, antigen_seq=antigen)
        df = pd.DataFrame([result])

        summary = (
            f"Binding probability: {result['binding_probability']:.4f}\n"
            f"Logit: {result['logit']:.4f}"
        )
        return summary, df

    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()


def load_bind_example():
    return EXAMPLE_HEAVY, EXAMPLE_ANTIGEN

def run_developability_ranking(
    target_name,
    heavy1, cdr31,
    heavy2, cdr32,
    heavy3, cdr33,
):
    try:
        candidates = []

        if str(heavy1).strip() and str(cdr31).strip():
            candidates.append({
                "candidate_name": "C1",
                "Target": target_name,
                "Heavy": heavy1,
                "cdr3": cdr31,
            })

        if str(heavy2).strip() and str(cdr32).strip():
            candidates.append({
                "candidate_name": "C2",
                "Target": target_name,
                "Heavy": heavy2,
                "cdr3": cdr32,
            })

        if str(heavy3).strip() and str(cdr33).strip():
            candidates.append({
                "candidate_name": "C3",
                "Target": target_name,
                "Heavy": heavy3,
                "cdr3": cdr33,
            })

        if not candidates:
            return (
                "Please provide at least one Heavy/CDRH3 candidate pair.",
                pd.DataFrame(),
                None,
            )

        scored_df = ranker.score_candidates(target_name=target_name, candidates=candidates)

        show_cols = [
            "candidate_name",
            "hard_filter_pass",
            "hard_filter_reasons",
            "developability_risk_score",
            "developability_risk_score_percentile",
            "heavy_nn_edit_distance",
            "cdr3_nn_edit_distance",
            "low_risk_claim",
            "high_diversity_claim",
            "overall_claim",
        ]
        show_cols = [c for c in show_cols if c in scored_df.columns]

        fig = ranker.plot_risk_distribution(
            target_name=target_name,
            scored_df=scored_df,
            out_path=str(OUTPUT_DIR / "developability.png"),
        )

        summary = (
            f"Target: {target_name}\n"
            f"Candidates evaluated: {len(scored_df)}\n"
            f"Developability ranking completed."
        )

        return summary, scored_df[show_cols], fig

    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame(), None


def load_developability_example():
    return (
        DEFAULT_TARGET,
        EXAMPLE_HEAVY,
        EXAMPLE_CDRH3,
        "",
        "",
        "",
        "",
    )

def run_full_pipeline(
    antigen,
    target_name,
    template_heavy,
    template_cdrh3,
    num_samples,
    min_len,
    sample_mode,
    temperature,
    deduplicate,
):
    try:
        if not antigen or not str(antigen).strip():
            return "Please provide an antigen sequence.", pd.DataFrame(), None

        if not template_heavy or not str(template_heavy).strip():
            return "Please provide a template heavy-chain sequence.", pd.DataFrame(), None

        if not template_cdrh3 or not str(template_cdrh3).strip():
            return "Please provide the template CDRH3 sequence.", pd.DataFrame(), None

        # 1) Generate candidates
        gen_df = generator.generate(
            antigen=antigen,
            num_samples=int(num_samples),
            min_len=int(min_len),
            sample_mode=sample_mode,
            temperature=float(temperature),
            deduplicate=bool(deduplicate),
        )

        if gen_df is None or len(gen_df) == 0:
            return "No candidates were generated.", pd.DataFrame(), None

        cdr3_col = detect_generated_cdr3_column(gen_df)

        # 2) Build heavy chains + predict binding
        binding_records = []
        dev_candidates = []

        for i, row in gen_df.iterrows():
            new_cdr3 = str(row[cdr3_col]).strip().upper()
            candidate_name = f"C{i+1}"

            full_heavy = graft_cdrh3_into_heavy(
                template_heavy=template_heavy,
                template_cdrh3=template_cdrh3,
                new_cdrh3=new_cdr3,
            )

            bind_result = binder.predict(
                heavy_seq=full_heavy,
                antigen_seq=antigen,
            )

            binding_records.append({
                "candidate_name": candidate_name,
                "generated_cdrh3": new_cdr3,
                "heavy_chain": full_heavy,
                "binding_probability": bind_result.get("binding_probability", None),
                "binding_logit": bind_result.get("logit", None),
            })

            dev_candidates.append({
                "candidate_name": candidate_name,
                "Target": target_name,
                "Heavy": full_heavy,
                "cdr3": new_cdr3,
            })

        bind_df = pd.DataFrame(binding_records)

        # 3) Developability
        dev_df = ranker.score_candidates(
            target_name=target_name,
            candidates=dev_candidates,
        )

        dev_fig = ranker.plot_risk_distribution(
            target_name=target_name,
            scored_df=dev_df,
            out_path=str(OUTPUT_DIR / "full_pipeline_developability.png"),
        )

        # 4) Merge
        merged = bind_df.merge(dev_df, on="candidate_name", how="left")

        # 5) Sort
        sort_cols = []
        ascending = []
        if "binding_probability" in merged.columns:
            sort_cols.append("binding_probability")
            ascending.append(False)
        if "developability_risk_score" in merged.columns:
            sort_cols.append("developability_risk_score")
            ascending.append(True)

        if sort_cols:
            merged = merged.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)

        out_csv = OUTPUT_DIR / "full_pipeline_results.csv"
        merged.to_csv(out_csv, index=False)

        summary = (
            f"Full pipeline completed.\n"
            f"Generated candidates: {len(gen_df)}\n"
            f"Target: {target_name}\n"
            f"Template heavy provided: yes\n"
            f"Saved results to: {out_csv}"
        )

        show_cols = [
            col for col in [
                "candidate_name",
                "generated_cdrh3",
                "binding_probability",
                "binding_logit",
                "hard_filter_pass",
                "hard_filter_reasons",
                "developability_risk_score",
                "developability_risk_score_percentile",
                "heavy_nn_edit_distance",
                "cdr3_nn_edit_distance",
                "low_risk_claim",
                "high_diversity_claim",
                "overall_claim",
            ]
            if col in merged.columns
        ]

        return summary, merged[show_cols], dev_fig

    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame(), None


def load_full_pipeline_example():
    return (
        EXAMPLE_ANTIGEN,
        DEFAULT_TARGET,
        EXAMPLE_HEAVY,
        EXAMPLE_CDRH3,
        16,
        8,
        "sample",
        1.0,
        True,
    )

with gr.Blocks(title="Antibody Design Application") as demo:
    gr.Markdown("""
# Antibody Design Application
Given an antigen sequence, the system can:
- Generate candidate CDRH3 sequences conditioned on the antigen  
- Predict antibody–antigen binding probability directly from sequence  
- Rank candidates based on developability-related sequence features  
- Run the full pipeline end-to-end: **Generate → Binding → Developability**
""")

    with gr.Tabs():

        with gr.Tab("Full pipeline"):
            gr.Markdown("""
### Module purpose
This mode runs the full closed-loop workflow in one click:
**Antigen sequence → generate CDRH3 candidates → graft into a heavy-chain template → predict binding → rank developability**
### Required inputs
- **Antigen sequence**
- **Target name** for developability ranking
- **Template heavy-chain sequence**
- **Template CDRH3 sequence** contained inside the template heavy chain
### Output
A ranked table of candidates and a developability risk-distribution plot.
""")

            with gr.Row():
                with gr.Column(scale=2):
                    full_antigen = gr.Textbox(
                        label="Antigen sequence",
                        lines=8,
                        placeholder="Paste antigen amino-acid sequence here...",
                    )

                    full_target = gr.Dropdown(
                        choices=AVAILABLE_TARGETS,
                        value=DEFAULT_TARGET,
                        label="Target name",
                    )

                    full_template_heavy = gr.Textbox(
                        label="Template heavy-chain sequence",
                        lines=5,
                        placeholder="Paste template heavy-chain sequence here...",
                    )

                    full_template_cdrh3 = gr.Textbox(
                        label="Template CDRH3 sequence",
                        lines=1,
                        placeholder="Paste template CDRH3 here...",
                    )

                    full_num_samples = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=16,
                        step=1,
                        label="Number of samples",
                    )

                    full_min_len = gr.Slider(
                        minimum=4,
                        maximum=20,
                        value=8,
                        step=1,
                        label="Minimum CDRH3 length",
                    )

                    full_sample_mode = gr.Dropdown(
                        choices=["sample", "argmax"],
                        value="sample",
                        label="Sampling mode",
                    )

                    full_temperature = gr.Slider(
                        minimum=0.2,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature",
                    )

                    full_deduplicate = gr.Checkbox(
                        value=True,
                        label="Remove duplicates",
                    )

                    with gr.Row():
                        full_run_btn = gr.Button("Run Full Pipeline")
                        full_example_btn = gr.Button("Load example")

                with gr.Column(scale=3):
                    full_summary = gr.Textbox(label="Pipeline summary", lines=5)
                    full_table = gr.Dataframe(label="Full pipeline results")
                    full_plot = gr.Plot(label="Developability risk distribution")

            full_run_btn.click(
                fn=run_full_pipeline,
                inputs=[
                    full_antigen,
                    full_target,
                    full_template_heavy,
                    full_template_cdrh3,
                    full_num_samples,
                    full_min_len,
                    full_sample_mode,
                    full_temperature,
                    full_deduplicate,
                ],
                outputs=[full_summary, full_table, full_plot],
            )

            full_example_btn.click(
                fn=load_full_pipeline_example,
                inputs=[],
                outputs=[
                    full_antigen,
                    full_target,
                    full_template_heavy,
                    full_template_cdrh3,
                    full_num_samples,
                    full_min_len,
                    full_sample_mode,
                    full_temperature,
                    full_deduplicate,
                ],
            )

        
        with gr.Tab("Generate"):
            gr.Markdown("""
### Module purpose
This module generates candidate **CDRH3 sequences** conditioned on an antigen sequence.
### Model background
It is based on an antigen-conditioned variational autoencoder.
### Input parameters
- **Number of samples**: how many candidate CDRH3 sequences to generate in one run.
- **Minimum CDRH3 length**: lower bound on generated CDRH3 length.
- **Sampling mode**:
  - `sample`: more diverse sequences.
  - `argmax`: more stable, less diverse.
- **Temperature**:
  - lower than 1.0: more conservative
  - around 1.0: default
  - higher than 1.0: more diverse
- **Remove duplicates**: whether to remove repeated generated CDRH3 sequences.
""")

            with gr.Row():
                with gr.Column(scale=2):
                    antigen_generate = gr.Textbox(
                        label="Antigen sequence",
                        lines=8,
                        placeholder="Paste antigen amino-acid sequence here...",
                    )

                    num_samples = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=32,
                        step=1,
                        label="Number of samples",
                    )

                    min_len = gr.Slider(
                        minimum=4,
                        maximum=20,
                        value=8,
                        step=1,
                        label="Minimum CDRH3 length",
                    )

                    sample_mode = gr.Dropdown(
                        choices=["sample", "argmax"],
                        value="sample",
                        label="Sampling mode",
                    )

                    temperature = gr.Slider(
                        minimum=0.2,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature",
                    )

                    deduplicate = gr.Checkbox(
                        value=True,
                        label="Remove duplicates",
                    )

                    with gr.Row():
                        generate_btn = gr.Button("Generate")
                        generate_example_btn = gr.Button("Load example")

                with gr.Column(scale=3):
                    generate_summary = gr.Textbox(label="Run summary", lines=4)
                    generate_table = gr.Dataframe(label="Generated CDRH3 candidates")

            generate_btn.click(
                fn=run_generation,
                inputs=[
                    antigen_generate,
                    num_samples,
                    min_len,
                    sample_mode,
                    temperature,
                    deduplicate,
                ],
                outputs=[generate_summary, generate_table],
            )

            generate_example_btn.click(
                fn=load_generate_example,
                inputs=[],
                outputs=[
                    antigen_generate,
                    num_samples,
                    min_len,
                    sample_mode,
                    temperature,
                    deduplicate,
                ],
            )

        with gr.Tab("Interaction prediction"):
            gr.Markdown("""
### Module purpose
This module predicts the probability that an antibody heavy chain interacts with an antigen sequence.
### Model background
It uses a protein language model encoder combined with bidirectional cross-attention to model sequence-level interactions between antibody and antigen.
### Input parameters
- **Heavy-chain sequence**: full antibody heavy-chain variable-region sequence.
- **Antigen sequence**: target antigen amino-acid sequence.
""")

            with gr.Row():
                with gr.Column(scale=2):
                    heavy_input = gr.Textbox(
                        label="Heavy-chain sequence",
                        lines=6,
                        placeholder="Paste heavy-chain amino-acid sequence here...",
                    )

                    antigen_bind = gr.Textbox(
                        label="Antigen sequence",
                        lines=8,
                        placeholder="Paste antigen amino-acid sequence here...",
                    )

                    with gr.Row():
                        bind_btn = gr.Button("Predict binding")
                        bind_example_btn = gr.Button("Load example")

                with gr.Column(scale=3):
                    bind_summary = gr.Textbox(label="Prediction summary", lines=4)
                    bind_table = gr.Dataframe(label="Prediction result")

            bind_btn.click(
                fn=run_binding_prediction,
                inputs=[heavy_input, antigen_bind],
                outputs=[bind_summary, bind_table],
            )

            bind_example_btn.click(
                fn=load_bind_example,
                inputs=[],
                outputs=[heavy_input, antigen_bind],
            )
        with gr.Tab("Developability"):
            gr.Markdown("""
### Module purpose
This module evaluates antibody candidates using **sequence-based developability criteria** and ranks them relative to antibodies associated with the same antigen.
""")

            with gr.Row():
                with gr.Column(scale=2):
                    target_dropdown = gr.Dropdown(
                        choices=AVAILABLE_TARGETS,
                        value=DEFAULT_TARGET,
                        label="Target name",
                    )

                    gr.Markdown("### Candidate C1")
                    heavy1 = gr.Textbox(label="Heavy 1", lines=4)
                    cdr31 = gr.Textbox(label="CDRH3 1", lines=1)

                    gr.Markdown("### Candidate C2")
                    heavy2 = gr.Textbox(label="Heavy 2", lines=4)
                    cdr32 = gr.Textbox(label="CDRH3 2", lines=1)

                    gr.Markdown("### Candidate C3")
                    heavy3 = gr.Textbox(label="Heavy 3", lines=4)
                    cdr33 = gr.Textbox(label="CDRH3 3", lines=1)

                    with gr.Row():
                        dev_btn = gr.Button("Rank developability")
                        dev_example_btn = gr.Button("Load example")

                with gr.Column(scale=3):
                    dev_summary = gr.Textbox(label="Developability summary", lines=4)
                    dev_table = gr.Dataframe(label="Developability ranking")
                    dev_plot = gr.Plot(label="Risk distribution")

            dev_btn.click(
                fn=run_developability_ranking,
                inputs=[
                    target_dropdown,
                    heavy1, cdr31,
                    heavy2, cdr32,
                    heavy3, cdr33,
                ],
                outputs=[dev_summary, dev_table, dev_plot],
            )

            dev_example_btn.click(
                fn=load_developability_example,
                inputs=[],
                outputs=[
                    target_dropdown,
                    heavy1, cdr31,
                    heavy2, cdr32,
                    heavy3, cdr33,
                ],
            )


demo.launch(server_name="127.0.0.1", server_port=7860)
