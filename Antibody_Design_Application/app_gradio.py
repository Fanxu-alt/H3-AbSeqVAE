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
DEFAULT_TARGET = EXAMPLE_TARGET if EXAMPLE_TARGET in AVAILABLE_TARGETS else (AVAILABLE_TARGETS[0] if AVAILABLE_TARGETS else None)


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

        fig = ranker.plot_risk_distribution(
            target_name=target_name,
            scored_df=scored_df,
            out_path=str(OUTPUT_DIR / "developability_panel_B.png"),
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

with gr.Blocks(title="Antibody Design Application") as demo:
    gr.Markdown("# Antibody Design Application")
    gr.Markdown(
        "Local tool for antigen-conditioned CDRH3 generation, antibody-antigen binding prediction, "
        "and developability-aware ranking."
    )

    with gr.Tabs():

        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Input")
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
                    gr.Markdown("### Output")
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
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Input")
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
                    gr.Markdown("### Output")
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
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Input")
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
                    gr.Markdown("### Output")
                    dev_summary = gr.Textbox(label="Developability summary", lines=4)
                    dev_table = gr.Dataframe(label="Developability ranking")
                    dev_plot = gr.Plot(label="Risk distribution (Panel B)")

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
