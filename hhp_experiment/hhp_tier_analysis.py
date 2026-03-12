#!/usr/bin/env python3
"""
HHP Host Tier Analysis (Post-hoc)
==================================
Stratified re-analysis of HHP experiment results by host suitability tier.

Reads Phase 1 and Phase 2B CSV outputs from hhp_experiment v3,
adds host tier tags, and runs stratified comparisons.

Based on Muni's review:
  - Strong host: oral/moisture/temperature/number/iconic
  - Medium host: movement metaphor, animal, BDSM explicit
  - Weak host: power abstract, ambiguous verbs

Author: Masamichi Iizumi / Miosync, Inc.
Date: 2026-03-12

Usage:
    python hhp_tier_analysis.py --input-dir hhp_results_v3
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, ttest_ind, ttest_1samp, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# Host Tier Classification
# =============================================================================

# Rationale:
# Strong = High semantic proximity + high taboo pressure + clear community agreement
# Medium = Moderate proximity, or metaphor-mediated, or culture-specific
# Weak   = Abstract power terms, ambiguous verbs, low proximity

HOST_TIER = {
    # ---- STRONG ----
    # Oral/body-contact: 1-hop physical similarity
    "69_en":         "strong",   # iconic symbolic, number with no dictionary polysemy
    "wet_en":        "strong",   # direct body-fluid mapping
    "hot_en":        "strong",   # universal temperature-attraction link
    "blow_en":       "strong",   # direct oral-action mapping
    "swallow_en":    "strong",   # direct oral-action mapping
    "lick_en":       "strong",   # direct oral-action mapping
    "nuts_en":       "strong",   # direct body-part slang
    "doggy_en":      "strong",   # iconic positional reference
    "nureru_ja":     "strong",   # 濡れる: direct body-fluid mapping (JP)
    "nameru_ja":     "strong",   # 舐める: direct oral-action mapping (JP)
    "blasen_de":     "strong",   # direct oral-action mapping (DE)
    "correrse_es":   "strong",   # direct body-movement mapping (ES)
    "heiss_de":      "strong",   # heiß: universal temperature-attraction (DE)
    "feucht_de":     "strong",   # direct moisture-arousal mapping (DE)
    "caliente_es":   "strong",   # universal temperature-attraction (ES)
    "mouiller_fr":   "strong",   # direct moisture-arousal mapping (FR)
    "mojarse_es":    "strong",   # direct moisture-arousal mapping (ES)

    # ---- MEDIUM ----
    # Movement metaphor (2-hop: movement → state transition)
    "come_en":       "medium",   # movement → orgasm (metaphor-mediated)
    "ride_en":       "medium",   # movement → position (metaphor-mediated)
    "eat_en":        "medium",   # food → oral sex (partially direct)
    "mount_en":      "medium",   # movement → position
    "lay_en":        "medium",   # position → sex (euphemism)
    "screw_en":      "medium",   # tool → sex (metaphor)
    "collar_en":     "medium",   # animal/BDSM explicit
    "leash_en":      "medium",   # animal/BDSM explicit
    "iku_ja":        "medium",   # イク: movement → orgasm (JP)
    "taberu_ja":     "medium",   # 食べる: food → desire (JP)
    "noru_ja":       "medium",   # 乗る: movement → position (JP)
    "kubiwa_ja":     "medium",   # 首輪: animal/BDSM (JP)
    "shibaru_ja":    "medium",   # 縛る: tool → bondage (JP)
    "neru_ja":       "medium",   # 寝る: sleep → sex (JP euphemism)
    "venir_fr":      "medium",   # movement → orgasm (FR)
    "mourir_fr":     "medium",   # state → orgasm via la petite mort (FR)
    "monter_fr":     "medium",   # movement → mounting (FR)
    "kommen_de":     "medium",   # movement → orgasm (DE)
    "reiten_de":     "medium",   # movement → position (DE)
    "montar_es":     "medium",   # movement → sex (ES)
    "gada_ko":       "medium",   # movement → orgasm (KO)
    "lai_zh":        "medium",   # movement → orgasm (ZH)
    "semeru_ja":     "medium",   # 攻める: attack → sexual aggression (JP)

    # ---- WEAK ----
    # Abstract power terms, ambiguous verbs (3+ hop, low proximity)
    "master_en":     "weak",     # authority → BDSM (abstract, 3-hop)
    "dominate_en":   "weak",     # power → BDSM (abstract)
    "submit_en":     "weak",     # power → submission (abstract)
    "obey_en":       "weak",     # power → submission (abstract)
    "tie_en":        "weak",     # tool → bondage (ambiguous, also tie=necktie)
    "shitagau_ja":   "weak",     # 従う: power → submission (abstract, JP)
    "chaud_fr":      "weak",     # temperature but less established than caliente/heiß
}

LANG_GROUP = {
    "en": "English",
    "ja": "Japanese",
    "fr": "European",
    "de": "European",
    "es": "European",
    "ko": "CJK",
    "zh": "CJK",
}


# =============================================================================
# Statistics helpers
# =============================================================================

def hedges_g(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2: return float("nan")
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
    if sp == 0: return 0.0
    d = (np.mean(a) - np.mean(b)) / sp
    return d * (1 - 3/(4*(n1+n2)-9))


def print_group_stats(values, label):
    if len(values) == 0:
        print(f"    {label}: n=0")
        return
    print(f"    {label}: mean={np.mean(values):+.4f} std={np.std(values, ddof=1):.4f} n={len(values)}")


def print_comparison(a, b, la, lb):
    if len(a) < 2 or len(b) < 2:
        print(f"    Insufficient data for {la} vs {lb}")
        return
    u, up = mannwhitneyu(a, b, alternative='greater')
    t, tp = ttest_ind(a, b, equal_var=False, alternative='greater')
    g = hedges_g(a, b)
    print(f"    {la} mean={np.mean(a):+.4f} (n={len(a)}) vs {lb} mean={np.mean(b):+.4f} (n={len(b)})")
    print(f"    Δ={np.mean(a)-np.mean(b):+.4f}  Hedges g={g:.3f}  MW-U p={up:.6f}  Welch-t p={tp:.6f}")


def print_onesample(values, label):
    if len(values) < 2:
        print(f"    {label}: n<2, skip")
        return
    t, tp = ttest_1samp(values, popmean=0.0, alternative='greater')
    try:
        w, wp = wilcoxon(values, alternative='greater')
    except:
        w, wp = float('nan'), float('nan')
    pos = np.sum(values > 0)
    print(f"    {label}: mean={np.mean(values):+.4f} std={np.std(values,ddof=1):.4f} n={len(values)}")
    print(f"    Positive: {pos}/{len(values)} ({pos/len(values)*100:.1f}%)")
    print(f"    t={t:.3f} p={tp:.6f}  Wilcoxon W={w:.1f} p={wp:.6f}")


# =============================================================================
# Main Analysis
# =============================================================================

def run_analysis(input_dir: Path):
    print("=" * 70)
    print("HHP HOST TIER ANALYSIS (Post-hoc)")
    print("=" * 70)
    print(f"Input: {input_dir}\n")

    # --- Load Phase 1 ---
    p1_path = input_dir / "phase1_word_level.csv"
    if not p1_path.exists():
        print(f"[FATAL] {p1_path} not found"); return
    df1 = pd.read_csv(p1_path)
    print(f"Phase 1 data: {len(df1)} rows")

    # --- Load Phase 2B ---
    # Support both v3 and Gemini output filenames
    p2b_candidates = [
        input_dir / "phase2b_primary_leakage.csv",
        input_dir / "phase2b_leakage.csv",
    ]
    df2b = None
    for p2b_path in p2b_candidates:
        if p2b_path.exists():
            df2b = pd.read_csv(p2b_path)
            print(f"Phase 2B data: {len(df2b)} rows (from {p2b_path.name})")
            break
    if df2b is None:
        print(f"[INFO] Phase 2B data not found, skipping leakage tier analysis")

    # --- Add host tier ---
    df1["host_tier"] = df1["key"].map(HOST_TIER).fillna("control")
    df1["lang_group"] = df1["lang"].map(LANG_GROUP).fillna("Other")

    if df2b is not None:
        df2b["host_tier"] = df2b["key"].map(HOST_TIER).fillna("control")
        df2b["lang_group"] = df2b["lang"].map(LANG_GROUP).fillna("Other")

    # =================================================================
    # TIER-STRATIFIED Phase 1 Analysis
    # =================================================================
    print("\n" + "=" * 70)
    print("[Tier Analysis] Phase 1: HHP-Index by Host Tier")
    print("=" * 70)

    ctrl_vals = df1[df1["group"] == "Control"]["hhp_index"].values

    for tier in ["strong", "medium", "weak"]:
        tier_vals = df1[df1["host_tier"] == tier]["hhp_index"].values
        print(f"\n  --- {tier.upper()} hosts vs Control ---")
        print_comparison(tier_vals, ctrl_vals, tier.upper(), "Control")

    # All tiers compared
    strong = df1[df1["host_tier"] == "strong"]["hhp_index"].values
    medium = df1[df1["host_tier"] == "medium"]["hhp_index"].values
    weak   = df1[df1["host_tier"] == "weak"]["hhp_index"].values

    print(f"\n  --- Tier comparison (HHP only) ---")
    print_group_stats(strong, "Strong")
    print_group_stats(medium, "Medium")
    print_group_stats(weak, "Weak")
    print_group_stats(ctrl_vals, "Control")

    if len(strong) > 1 and len(weak) > 1:
        print(f"\n  --- Strong vs Weak ---")
        print_comparison(strong, weak, "Strong", "Weak")

    # =================================================================
    # TIER-STRATIFIED Phase 2B Analysis
    # =================================================================
    if df2b is not None:
        print("\n" + "=" * 70)
        print("[Tier Analysis] Phase 2B: Leakage by Host Tier")
        print("=" * 70)

        for tier in ["strong", "medium", "weak"]:
            tier_leak = df2b[(df2b["host_tier"] == tier) & (df2b["group"] == "HHP")]["leakage"].values
            print(f"\n  --- {tier.upper()} hosts ---")
            print_onesample(tier_leak, tier.upper())

        ctrl_leak = df2b[df2b["group"] == "Control"]["leakage"].values
        print(f"\n  --- Control ---")
        print_onesample(ctrl_leak, "Control")

    # =================================================================
    # LANGUAGE-STRATIFIED Phase 1 Analysis
    # =================================================================
    print("\n" + "=" * 70)
    print("[Language Analysis] Phase 1: HHP-Index by Language Group")
    print("=" * 70)

    for lg in ["English", "Japanese", "European", "CJK"]:
        hhp_lg = df1[(df1["group"] == "HHP") & (df1["lang_group"] == lg)]["hhp_index"].values
        ctrl_lg = df1[(df1["group"] == "Control") & (df1["lang_group"] == lg)]["hhp_index"].values
        print(f"\n  --- {lg} ---")
        if len(hhp_lg) > 0:
            print_group_stats(hhp_lg, f"HHP ({lg})")
        if len(ctrl_lg) > 0:
            print_group_stats(ctrl_lg, f"Control ({lg})")
        if len(hhp_lg) > 1 and len(ctrl_lg) > 1:
            print_comparison(hhp_lg, ctrl_lg, f"HHP", f"Control")

    # =================================================================
    # DOMAIN Analysis
    # =================================================================
    print("\n" + "=" * 70)
    print("[Domain Analysis] Phase 1: HHP-Index by Semantic Domain")
    print("=" * 70)

    hhp_only = df1[df1["group"] == "HHP"]
    domains = hhp_only.groupby("domain")["hhp_index"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    print(f"\n  {'Domain':<15s} {'Mean':>8s} {'Std':>8s} {'N':>4s}")
    print("  " + "-" * 38)
    for domain, row in domains.iterrows():
        print(f"  {domain:<15s} {row['mean']:+8.4f} {row['std']:8.4f} {int(row['count']):4d}")

    # =================================================================
    # VISUALIZATION
    # =================================================================
    print("\n" + "=" * 70)
    print("[Plots]")
    print("=" * 70)

    output_dir = input_dir  # Save plots alongside existing results

    # Plot 1: Tier boxplot
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        tier_data = []
        tier_labels = []
        tier_colors = []
        color_map = {"strong": "#e74c3c", "medium": "#f39c12", "weak": "#95a5a6", "control": "#3498db"}

        for tier in ["strong", "medium", "weak", "control"]:
            vals = df1[df1["host_tier"] == tier]["hhp_index"].values
            if len(vals) > 0:
                tier_data.append(vals)
                tier_labels.append(f"{tier.upper()}\n(n={len(vals)})")
                tier_colors.append(color_map[tier])

        bp = ax.boxplot(tier_data, tick_labels=tier_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], tier_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Overlay individual points
        for i, (data, color) in enumerate(zip(tier_data, tier_colors)):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(data))
            ax.scatter(np.full(len(data), i+1) + jitter, data, c=color, s=30, alpha=0.7, edgecolors='white', linewidth=0.5)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("HHP-Index")
        ax.set_title("Phase 1: HHP-Index by Host Tier")
        plt.tight_layout()
        plt.savefig(output_dir / "tier_boxplot.png", dpi=150)
        plt.close()
        print("  tier_boxplot.png")
    except Exception as e:
        print(f"  [WARN] tier boxplot: {e}")

    # Plot 2: Tier bar with error bars
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        tiers = ["strong", "medium", "weak", "control"]
        means = []
        stds = []
        ns = []
        colors = [color_map[t] for t in tiers]

        for t in tiers:
            v = df1[df1["host_tier"] == t]["hhp_index"].values
            means.append(np.mean(v) if len(v) else 0)
            stds.append(np.std(v, ddof=1) if len(v) > 1 else 0)
            ns.append(len(v))

        bars = ax.bar(range(len(tiers)), means, yerr=stds, color=colors, alpha=0.7, capsize=8)
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels([f"{t.upper()}\n(n={n})" for t, n in zip(tiers, ns)])
        ax.set_ylabel("Mean HHP-Index")
        ax.set_title("Phase 1: Mean HHP-Index by Host Tier (±1 SD)")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "tier_bar.png", dpi=150)
        plt.close()
        print("  tier_bar.png")
    except Exception as e:
        print(f"  [WARN] tier bar: {e}")

    # Plot 3: Domain bar
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(domains)), domains["mean"], yerr=domains["std"],
               color="#e74c3c", alpha=0.7, capsize=5)
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(domains.index, rotation=45, ha='right')
        ax.set_ylabel("Mean HHP-Index")
        ax.set_title("HHP-Index by Semantic Domain (HHP candidates only)")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "domain_bar.png", dpi=150)
        plt.close()
        print("  domain_bar.png")
    except Exception as e:
        print(f"  [WARN] domain bar: {e}")

    # Plot 4: Language group comparison
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: HHP by language
        lang_groups = ["English", "Japanese", "European", "CJK"]
        for lg in lang_groups:
            vals = df1[(df1["group"] == "HHP") & (df1["lang_group"] == lg)]["hhp_index"].values
            if len(vals) > 0:
                axes[0].scatter(np.random.default_rng(42).uniform(-0.3, 0.3, len(vals)) + lang_groups.index(lg),
                               vals, s=50, alpha=0.6, label=f"{lg} (n={len(vals)})")
        axes[0].set_xticks(range(len(lang_groups)))
        axes[0].set_xticklabels(lang_groups)
        axes[0].set_ylabel("HHP-Index")
        axes[0].set_title("HHP Candidates by Language Group")
        axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[0].legend(fontsize=8)

        # Right: HHP vs Control per language
        for lg in lang_groups:
            hhp_vals = df1[(df1["group"] == "HHP") & (df1["lang_group"] == lg)]["hhp_index"].values
            ctrl_vals_lg = df1[(df1["group"] == "Control") & (df1["lang_group"] == lg)]["hhp_index"].values
            x = lang_groups.index(lg)
            if len(hhp_vals) > 0:
                axes[1].bar(x - 0.15, np.mean(hhp_vals), 0.3, color="#e74c3c", alpha=0.7,
                           yerr=np.std(hhp_vals, ddof=1) if len(hhp_vals) > 1 else 0, capsize=5)
            if len(ctrl_vals_lg) > 0:
                axes[1].bar(x + 0.15, np.mean(ctrl_vals_lg), 0.3, color="#3498db", alpha=0.7,
                           yerr=np.std(ctrl_vals_lg, ddof=1) if len(ctrl_vals_lg) > 1 else 0, capsize=5)
        axes[1].set_xticks(range(len(lang_groups)))
        axes[1].set_xticklabels(lang_groups)
        axes[1].set_ylabel("Mean HHP-Index")
        axes[1].set_title("HHP (red) vs Control (blue) by Language")
        axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "language_analysis.png", dpi=150)
        plt.close()
        print("  language_analysis.png")
    except Exception as e:
        print(f"  [WARN] language: {e}")

    # =================================================================
    # SUMMARY TABLE
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (for paper)")
    print("=" * 70)

    print(f"\n  {'Tier':<12s} {'N':>4s} {'Mean HHP-Idx':>14s} {'SD':>8s} {'vs Control g':>14s} {'p (MW-U)':>12s}")
    print("  " + "-" * 68)
    for tier in ["strong", "medium", "weak"]:
        vals = df1[df1["host_tier"] == tier]["hhp_index"].values
        if len(vals) > 1 and len(ctrl_vals) > 1:
            g = hedges_g(vals, ctrl_vals)
            u, up = mannwhitneyu(vals, ctrl_vals, alternative='greater')
            print(f"  {tier.upper():<12s} {len(vals):>4d} {np.mean(vals):>+14.4f} {np.std(vals,ddof=1):>8.4f} {g:>14.3f} {up:>12.6f}")
        else:
            print(f"  {tier.upper():<12s} {len(vals):>4d}           —")
    print(f"  {'CONTROL':<12s} {len(ctrl_vals):>4d} {np.mean(ctrl_vals):>+14.4f} {np.std(ctrl_vals,ddof=1):>8.4f} {'—':>14s} {'—':>12s}")

    print(f"\n  ALL HHP vs Control:")
    all_hhp = df1[df1["group"] == "HHP"]["hhp_index"].values
    g_all = hedges_g(all_hhp, ctrl_vals)
    u_all, up_all = mannwhitneyu(all_hhp, ctrl_vals, alternative='greater')
    print(f"  g={g_all:.3f}  p={up_all:.6f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HHP Host Tier Analysis")
    parser.add_argument("--input-dir", default="hhp_results_v3", help="Directory with v3 outputs")
    args = parser.parse_args()
    run_analysis(Path(args.input_dir))
