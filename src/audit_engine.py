"""
Core foundation model evaluation engine.
Extracts token surprisal and hidden-state projection across historical semantic benchmarks.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import aggregate_audit_scores, compute_token_surprisal, cosine_similarity

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("anachronism_auditor")


class AnachronismAuditor:
    def __init__(self, model_name: str = "gpt2", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading checkpoint '{model_name}' on device '{self.device}'...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _extract_pooled_repr(self, text: str) -> np.ndarray:
        """Runs a forward pass and extracts mean-pooled final hidden states."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Last hidden layer: shape (batch_size, seq_len, hidden_dim)
            last_hidden = outputs.hidden_states[-1][0]
            # Use attention mask to prevent pooling over padding tokens
            mask = inputs["attention_mask"][0].unsqueeze(-1)
            summed = torch.sum(last_hidden * mask, dim=0)
            counts = torch.clamp(mask.sum(dim=0), min=1.0)
            pooled = (summed / counts).detach().cpu().numpy()
        return pooled

    def get_concept_centroid(self, phrases: List[str]) -> np.ndarray:
        """Averages contextual representations across reference phrases."""
        vecs = [self._extract_pooled_repr(p) for p in phrases]
        mean_vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        return mean_vec / norm if norm > 0 else mean_vec

    def audit_entry(self, entry: Dict) -> Dict:
        clause = entry["clause"]
        target_token = entry["token"].lower()

        # Tokenize and identify exact target token position in input sequence
        inputs = self.tokenizer(clause, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0].tolist()
        tokens = [self.tokenizer.decode([t_id]).strip().lower() for t_id in input_ids]

        target_idx = -1
        for idx, t in enumerate(tokens):
            if target_token in t or t in target_token:
                target_idx = idx
                break

        # If subword splitting obscured it, fall back to last non-punct token
        if target_idx <= 0:
            target_idx = len(input_ids) - 1

        # Forward pass for conditional probability and representations
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0].detach().cpu().numpy()

        # Token surprisal evaluated at target position
        logits_at_step = logits[target_idx - 1]
        surprisal = compute_token_surprisal(input_ids[target_idx], logits_at_step)

        # Clause contextual representation
        clause_rep = self._extract_pooled_repr(clause)

        # Historical vs modern reference centroids
        c_hist = self.get_concept_centroid(entry["historical_domain"])
        c_mod = self.get_concept_centroid(entry["modern_distractor"])

        sim_hist = cosine_similarity(clause_rep, c_hist)
        sim_mod = cosine_similarity(clause_rep, c_mod)
        delta = sim_mod - sim_hist
        is_anachronistic = bool(sim_mod > sim_hist)

        return {
            "id": entry.get("id", "N/A"),
            "token": entry["token"],
            "surprisal_bits": round(surprisal, 3),
            "sim_historical": round(sim_hist, 4),
            "sim_modern": round(sim_mod, 4),
            "delta_projection": round(delta, 4),
            "is_anachronistic": is_anachronistic,
            "is_statutory_anchor": entry.get("is_statutory_anchor", False),
        }


def run_evaluation(data_path: str, model_name: str, output_plot: str):
    if not os.path.exists(data_path):
        logger.error(f"Benchmark file not found: {data_path}")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    logger.info(f"Loaded {len(benchmark)} test cases from '{data_path}'")
    auditor = AnachronismAuditor(model_name=model_name)

    results = []
    logger.info("Executing benchmark inference...")
    for idx, item in enumerate(benchmark, start=1):
        res = auditor.audit_entry(item)
        results.append(res)
        logger.info(
            f"[{idx}/{len(benchmark)}] token={res['token']:<14} "
            f"surprisal={res['surprisal_bits']:<6.2f} "
            f"hist_sim={res['sim_historical']:<6.4f} "
            f"mod_sim={res['sim_modern']:<6.4f} "
            f"status={'FAIL' if res['is_anachronistic'] else 'PASS'}"
        )

    aer, mean_delta, mean_surprisal = aggregate_audit_scores(results)

    logger.info("=" * 60)
    logger.info(f"Benchmark Summary for '{model_name}':")
    logger.info(f"  Anachronistic Error Rate (AER): {aer:.2%}")
    logger.info(f"  Mean Delta Projection (Mod - Hist): {mean_delta:+.4f}")
    logger.info(f"  Mean Target Surprisal: {mean_surprisal:.2f} bits")
    logger.info("=" * 60)

    # Plot empirical breakdown
    os.makedirs(os.path.dirname(output_plot) or ".", exist_ok=True)
    tokens = [r["token"] for r in results]
    hist_sims = [r["sim_historical"] for r in results]
    mod_sims = [r["sim_modern"] for r in results]

    x = np.arange(len(tokens))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    ax.bar(x - bar_width / 2, hist_sims, bar_width, label="Historical Ground Truth", color="#1f4e5b", edgecolor="black", linewidth=0.5)
    ax.bar(x + bar_width / 2, mod_sims, bar_width, label="Modern Colloquial Bias", color="#b33939", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Cosine Projection Alignment", fontsize=10)
    ax.set_title(f"Model: {model_name} | Empirical Anachronistic Error Rate (AER = {aer:.1%})", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, fontsize=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_plot)
    logger.info(f"Saved visualization artifact to '{output_plot}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit LLM semantic projection against historical statutory ground truth.")
    parser.add_argument("--data", type=str, default="data/benchmark_statutory.json", help="Path to ground truth benchmark JSON.")
    parser.add_argument("--model", type=str, default="gpt2", help="Hugging Face model identifier or local directory.")
    parser.add_argument("--output", type=str, default="output/anachronism_divergence.png", help="Path for output figure.")
    args = parser.parse_args()

    run_evaluation(args.data, args.model, args.output)
