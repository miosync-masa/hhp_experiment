#!/usr/bin/env python3
"""
HHP Cross-Validation: Gemini Embedding 2
==========================================
Runs the same HHP analysis using Google's Gemini Embedding 2 model
for cross-model validation.

If HHP patterns replicate across OpenAI text-embedding-3-large AND
Google gemini-embedding-2-preview, the patterns are model-independent
and reflect structure in language itself, not model-specific artifacts.

Requirements:
    pip install google-generativeai numpy pandas matplotlib scikit-learn scipy python-dotenv

Usage:
    export GOOGLE_API_KEY="your-key-here"
    python hhp_gemini_cross.py --output-dir hhp_results_gemini

    Then compare with OpenAI results:
    python hhp_cross_compare.py --openai-dir hhp_results_v3 --gemini-dir hhp_results_gemini

Author: Masamichi Iizumi / Miosync, Inc.
Date: 2026-03-12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, ttest_ind, ttest_1samp, wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
load_dotenv()

# Import the same data inventory from v3
# We duplicate the essential data structures here for standalone execution
# (in production, these would be imported from a shared module)

GEMINI_MODEL = "gemini-embedding-2-preview"
DEFAULT_OUTPUT_DIR = "hhp_results_gemini"
DEFAULT_CACHE_FILE = "gemini_embedding_cache.json"
SEED = 42

# =============================================================================
# Google GenAI Client
# =============================================================================

try:
    import google.generativeai as genai
except ImportError:
    print("[FATAL] google-generativeai not installed.")
    print("  pip install google-generativeai")
    sys.exit(1)


class GeminiEmbeddingRunner:
    """Wrapper for Gemini Embedding API with disk cache."""

    def __init__(self, cache_path: Path, api_delay: float = 0.1):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self.api_delay = api_delay
        self.cache_path = cache_path
        self.cache: Dict[str, List[float]] = {}
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.cache = raw.get("embeddings", {})
            print(f"  Loaded {len(self.cache)} cached embeddings")

    def embed_texts(self, texts: Sequence[str], verbose: bool = True) -> Dict[str, np.ndarray]:
        unique = list(dict.fromkeys(texts))
        result: Dict[str, np.ndarray] = {}
        pending: List[str] = []

        for t in unique:
            if t in self.cache:
                result[t] = np.array(self.cache[t])
            else:
                pending.append(t)

        if verbose:
            print(f"  Cached: {len(result)} / Pending: {len(pending)}")

        # Gemini supports batch embedding via embed_content with list input
        batch_size = 50  # Conservative batch size for Gemini
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            if verbose:
                print(f"  Requesting batch {start // batch_size + 1}: {len(batch)} texts")
            try:
                response = genai.embed_content(
                    model=f"models/{GEMINI_MODEL}",
                    content=batch,
                )
                # response.embedding is a list of lists when input is a list
                embeddings_list = response['embedding']
                for text, emb in zip(batch, embeddings_list):
                    vec = np.array(emb, dtype=float)
                    self.cache[text] = vec.tolist()
                    result[text] = vec
            except Exception as e:
                print(f"  [ERROR] Batch failed: {e}")
                # Fall back to individual embedding
                for text in batch:
                    try:
                        resp = genai.embed_content(
                            model=f"models/{GEMINI_MODEL}",
                            content=text,
                        )
                        vec = np.array(resp['embedding'], dtype=float)
                        self.cache[text] = vec.tolist()
                        result[text] = vec
                        time.sleep(self.api_delay)
                    except Exception as e2:
                        print(f"    [ERROR] '{text[:40]}': {e2}")

            self.save_cache()
            time.sleep(self.api_delay)

        return result

    def save_cache(self):
        payload = {
            "saved_at": datetime.now().isoformat(),
            "model": GEMINI_MODEL,
            "embeddings": self.cache,
        }
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)


# =============================================================================
# Import word sets and templates from the main experiment
# (Duplicated here for standalone execution)
# =============================================================================

# We import the data from the project file if available, otherwise define inline
def load_experiment_data():
    """Try to import from hhp_experiment, fall back to inline definitions."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from hhp_experiment import (
            HHP_CANDIDATES, CONTROL_WORDS,
            ANCHORS, build_context_templates, build_text_inventory,
            PHASE2B_BASELINES,
        )
        print("  Loaded data from hhp_experiment.py")
        return HHP_CANDIDATES, CONTROL_WORDS, ANCHORS, build_context_templates, build_text_inventory, PHASE2B_BASELINES
    except ImportError:
        print("  [FATAL] Cannot import hhp_experiment.py - place it in the same directory")
        sys.exit(1)


# =============================================================================
# Analysis functions (same as v3)
# =============================================================================

def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def cosine(a, b):
    return float(cosine_similarity([a], [b])[0][0])

def safe_std(values, ddof=1):
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=ddof)) if len(arr) > ddof else float("nan")

def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2: return float("nan")
    sp = np.sqrt(((n1-1)*np.var(a,ddof=1) + (n2-1)*np.var(b,ddof=1)) / (n1+n2-2))
    if sp == 0: return 0.0
    d = (np.mean(a) - np.mean(b)) / sp
    return d * (1 - 3/(4*(n1+n2)-9))

def centroid(embs, texts):
    vecs = [embs[t] for t in texts if t in embs]
    return l2_normalize(np.mean(vecs, axis=0)) if vecs else None

def print_summary(a, b, la="HHP", lb="Control"):
    if len(a) < 2 or len(b) < 2: return
    u, up = mannwhitneyu(a, b, alternative='greater')
    t, tp = ttest_ind(a, b, equal_var=False, alternative='greater')
    g = hedges_g(a, b)
    print(f"  {la}: mean={np.mean(a):+.4f} std={safe_std(a):.4f} n={len(a)}")
    print(f"  {lb}: mean={np.mean(b):+.4f} std={safe_std(b):.4f} n={len(b)}")
    print(f"  Δ={np.mean(a)-np.mean(b):+.4f}  g={g:.3f}  MW p={up:.6f}  t p={tp:.6f}")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_gemini_experiment(args):
    print("=" * 70)
    print("HHP CROSS-VALIDATION: Gemini Embedding 2")
    print("=" * 70)
    print(f"Model: {GEMINI_MODEL}")
    print(f"Time: {datetime.now().isoformat()}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data from main experiment
    print("[Phase 0] Loading experiment data...")
    HHP_CANDIDATES, CONTROL_WORDS, ANCHORS, build_context_templates_fn, build_text_inventory_fn, PHASE2B_BASELINES = load_experiment_data()

    hhp_templates, control_templates, matched_pairs = build_context_templates_fn()
    all_texts = build_text_inventory_fn(hhp_templates, control_templates, matched_pairs)
    print(f"  Total texts: {len(all_texts)}")

    # Initialize Gemini runner
    print(f"\n[Phase 0] Embedding with {GEMINI_MODEL}...")
    runner = GeminiEmbeddingRunner(
        cache_path=output_dir / args.cache_file,
        api_delay=0.2,  # Slightly higher delay for Gemini rate limits
    )
    embs_raw = runner.embed_texts(all_texts, verbose=True)

    # Normalize
    embs = {k: l2_normalize(v) for k, v in embs_raw.items()}
    print(f"  Embedded: {len(embs)}/{len(all_texts)}")

    # Save
    with open(output_dir / "raw_embeddings.json", "w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in embs.items()}, f, ensure_ascii=False)

    # Build centroids (language-aware, matching v3 logic)
    LANGS = ["en", "ja", "fr", "de", "es", "ko", "zh"]

    sex_centroids = {}
    neut_centroids = {}
    for lang in LANGS:
        sex_centroids[lang] = centroid(embs, ANCHORS["sexual"].get(lang, []))
        neut_centroids[lang] = centroid(embs, ANCHORS["neutral"].get(lang, []))
    global_sex = centroid(embs, [x for v in ANCHORS["sexual"].values() for x in v])
    global_neut = centroid(embs, [x for v in ANCHORS["neutral"].values() for x in v])

    primary_centroids = {}
    for domain, by_lang in ANCHORS["primary"].items():
        primary_centroids[domain] = {}
        for lang in LANGS:
            primary_centroids[domain][lang] = centroid(embs, by_lang.get(lang, []))

    def get_sex(lang):
        c = sex_centroids.get(lang)
        if c is not None:
            return c
        return global_sex
    def get_neut(lang):
        c = neut_centroids.get(lang)
        if c is not None:
            return c
        return global_neut
    def get_primary(domain, lang):
        c = primary_centroids.get(domain, {}).get(lang)
        if c is not None:
            return c
        return primary_centroids.get(domain, {}).get("en")

    # =================================================================
    # PHASE 1
    # =================================================================
    print("\n" + "=" * 70)
    print("[Phase 1] WORD-LEVEL ANALYSIS (Gemini)")
    print("=" * 70)

    rows = []
    for gn, wd in [("HHP", HHP_CANDIDATES), ("Control", CONTROL_WORDS)]:
        print(f"\n  --- {gn} ---")
        for k, info in wd.items():
            w = info["word"]
            if w not in embs: continue
            lang = info["lang"]
            sc = get_sex(lang); nc = get_neut(lang)
            if sc is None or nc is None: continue
            ss = cosine(embs[w], sc); sn = cosine(embs[w], nc)
            hi = ss - sn
            row = {"key": k, "word": w, "lang": lang, "domain": info["domain"],
                   "group": gn, "sim_sexual": ss, "sim_neutral": sn, "hhp_index": hi}
            pc = get_primary(info["domain"], lang)
            if pc is not None:
                row["sim_primary"] = cosine(embs[w], pc)
            rows.append(row)
            print(f"    {w:15s} ({lang}) HHP-Idx: {hi:+.4f}")

    df1 = pd.DataFrame(rows)
    df1.to_csv(output_dir / "phase1_word_level.csv", index=False, encoding="utf-8-sig")

    hhp_vals = df1[df1.group=="HHP"].hhp_index.values
    ctrl_vals = df1[df1.group=="Control"].hhp_index.values
    print(f"\n  === Phase 1 Summary (Gemini) ===")
    print_summary(hhp_vals, ctrl_vals)

    # =================================================================
    # PHASE 2B: Leakage
    # =================================================================
    print("\n" + "=" * 70)
    print("[Phase 2B] PRIMARY-CONTEXT LEAKAGE (Gemini)")
    print("=" * 70)

    rows2b = []
    for gn, templates_map, inventory in [
        ("HHP", hhp_templates, HHP_CANDIDATES),
        ("Control", control_templates, CONTROL_WORDS),
    ]:
        for key, templates in templates_map.items():
            primary_text = templates["primary"]
            baseline_text = PHASE2B_BASELINES.get(key)
            if baseline_text is None: continue
            if primary_text not in embs or baseline_text not in embs: continue
            info = inventory[key]
            lang = info["lang"]
            sc = get_sex(lang); nc = get_neut(lang)
            if sc is None or nc is None: continue

            pp = cosine(embs[primary_text], sc) - cosine(embs[primary_text], nc)
            bp = cosine(embs[baseline_text], sc) - cosine(embs[baseline_text], nc)
            leak = pp - bp

            rows2b.append({"key": key, "word": info["word"], "lang": lang,
                          "group": gn, "leakage": leak})
            print(f"  {info['word']:15s} ({lang}) leakage: {leak:+.4f}  {gn}")

    df2b = pd.DataFrame(rows2b)
    df2b.to_csv(output_dir / "phase2b_leakage.csv", index=False, encoding="utf-8-sig")

    hhp_leak = df2b[df2b.group=="HHP"].leakage.values
    ctrl_leak = df2b[df2b.group=="Control"].leakage.values

    print(f"\n  === Phase 2B HHP ===")
    if len(hhp_leak) > 1:
        t, tp = ttest_1samp(hhp_leak, 0, alternative='greater')
        try: w, wp = wilcoxon(hhp_leak, alternative='greater')
        except: w, wp = float('nan'), float('nan')
        print(f"  mean={np.mean(hhp_leak):+.4f} pos={np.sum(hhp_leak>0)}/{len(hhp_leak)}  t p={tp:.6f}  W p={wp:.6f}")

    print(f"\n  === Phase 2B Control ===")
    if len(ctrl_leak) > 1:
        t, tp = ttest_1samp(ctrl_leak, 0, alternative='greater')
        try: w, wp = wilcoxon(ctrl_leak, alternative='greater')
        except: w, wp = float('nan'), float('nan')
        print(f"  mean={np.mean(ctrl_leak):+.4f} pos={np.sum(ctrl_leak>0)}/{len(ctrl_leak)}  t p={tp:.6f}  W p={wp:.6f}")

    # =================================================================
    # PHASE 3b: 69 Triplet
    # =================================================================
    print("\n" + "=" * 70)
    print("[Phase 3b] 69 TRIPLET (Gemini)")
    print("=" * 70)

    sc_en = get_sex("en"); nc_en = get_neut("en")
    print(f"\n  {'Number':>6s} | {'sim_sex':>8s} | {'sim_neut':>8s} | {'pull':>8s}")
    print("  " + "-" * 42)
    for n in [42, 67, 68, 69, 70, 71]:
        sent = f"The answer to question 5 is {n}."
        if sent in embs and sc_en is not None and nc_en is not None:
            ss = cosine(embs[sent], sc_en); sn = cosine(embs[sent], nc_en)
            m = " <<<" if n == 69 else ""
            print(f"  {n:6d} | {ss:8.4f} | {sn:8.4f} | {ss-sn:+8.4f}{m}")

    # =================================================================
    # MATCHED PAIRS (Phase 3)
    # =================================================================
    print("\n" + "=" * 70)
    print("[Phase 3] MATCHED PAIRS (Gemini)")
    print("=" * 70)

    rows3 = []
    for pk, mp in matched_pairs.items():
        hs, cs = mp["hhp_sentence"], mp["control_sentence"]
        if hs not in embs or cs not in embs: continue
        lang = mp["lang"]
        sc = get_sex(lang); nc = get_neut(lang)
        if sc is None or nc is None: continue
        hp = cosine(embs[hs], sc) - cosine(embs[hs], nc)
        cp = cosine(embs[cs], sc) - cosine(embs[cs], nc)
        delta = hp - cp
        rows3.append({"pair": pk, "lang": lang, "hhp_pull": hp, "ctrl_pull": cp, "delta": delta})
        marker = ">>>" if delta > 0 else "   "
        print(f"  {marker} {pk:30s} ({lang}) Δ:{delta:+.4f}")

    df3 = pd.DataFrame(rows3)
    df3.to_csv(output_dir / "phase3_matched.csv", index=False, encoding="utf-8-sig")

    if len(df3) > 0:
        deltas = df3.delta.values
        print(f"\n  Mean Δ: {np.mean(deltas):+.4f}  Pos: {np.sum(deltas>0)}/{len(deltas)}")

    # =================================================================
    # PLOTS
    # =================================================================
    try:
        fig, ax = plt.subplots(figsize=(14, 10))
        d = df1.sort_values("hhp_index")
        colors = ["#e74c3c" if g=="HHP" else "#3498db" for g in d.group]
        ax.barh(range(len(d)), d.hhp_index, color=colors)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([f"{r.word} ({r.lang})" for r in d.itertuples()], fontsize=7)
        ax.set_xlabel("HHP-Index"); ax.set_title(f"Phase 1: HHP-Index ({GEMINI_MODEL})")
        ax.axvline(0, color="gray", ls="--", alpha=.5); plt.tight_layout()
        plt.savefig(output_dir / "phase1_hhp_index_gemini.png", dpi=150); plt.close()
        print("\n  phase1_hhp_index_gemini.png")
    except Exception as e:
        print(f"  [W] Plot: {e}")

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print(f"GEMINI CROSS-VALIDATION COMPLETE")
    print("=" * 70)
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name:45s} {f.stat().st_size:>10,} bytes")

    p1_pass = np.mean(hhp_vals) > np.mean(ctrl_vals) if len(hhp_vals) > 0 and len(ctrl_vals) > 0 else False
    print(f"\n  P1 HHP > Control: {'YES' if p1_pass else 'NO'}")
    print()


# =============================================================================
# Cross-Model Comparison Script
# =============================================================================

def run_cross_compare(openai_dir: Path, gemini_dir: Path):
    """Compare OpenAI and Gemini results side by side."""
    print("=" * 70)
    print("CROSS-MODEL COMPARISON: OpenAI vs Gemini")
    print("=" * 70)

    df_oai = pd.read_csv(openai_dir / "phase1_word_level.csv")
    df_gem = pd.read_csv(gemini_dir / "phase1_word_level.csv")

    # Merge on key
    merged = df_oai[["key", "word", "lang", "group", "hhp_index"]].merge(
        df_gem[["key", "hhp_index"]], on="key", suffixes=("_openai", "_gemini")
    )

    print(f"\n  Matched words: {len(merged)}")
    print(f"\n  {'Word':15s} {'Lang':>4s} {'Group':>8s} {'OpenAI':>10s} {'Gemini':>10s} {'Agree':>6s}")
    print("  " + "-" * 58)

    agree_count = 0
    for _, r in merged.iterrows():
        agree = (r.hhp_index_openai > 0) == (r.hhp_index_gemini > 0)
        if agree: agree_count += 1
        print(f"  {r.word:15s} {r.lang:>4s} {r.group:>8s} {r.hhp_index_openai:+10.4f} {r.hhp_index_gemini:+10.4f} {'✓' if agree else '✗':>6s}")

    print(f"\n  Sign agreement: {agree_count}/{len(merged)} ({agree_count/len(merged)*100:.1f}%)")

    # Correlation
    from scipy.stats import spearmanr, pearsonr
    r_pearson, p_pearson = pearsonr(merged.hhp_index_openai, merged.hhp_index_gemini)
    r_spearman, p_spearman = spearmanr(merged.hhp_index_openai, merged.hhp_index_gemini)
    print(f"\n  Pearson r: {r_pearson:.4f} (p={p_pearson:.6f})")
    print(f"  Spearman ρ: {r_spearman:.4f} (p={p_spearman:.6f})")

    # Group-level comparison
    for group in ["HHP", "Control"]:
        g = merged[merged.group == group]
        print(f"\n  --- {group} ---")
        print(f"  OpenAI mean: {g.hhp_index_openai.mean():+.4f}")
        print(f"  Gemini mean: {g.hhp_index_gemini.mean():+.4f}")

    # Scatter plot
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        for group, color in [("HHP", "#e74c3c"), ("Control", "#3498db")]:
            g = merged[merged.group == group]
            ax.scatter(g.hhp_index_openai, g.hhp_index_gemini, c=color, label=group, s=60, alpha=0.7)
            for _, r in g.iterrows():
                ax.annotate(r.word, (r.hhp_index_openai, r.hhp_index_gemini),
                           fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")

        lim = max(abs(merged.hhp_index_openai).max(), abs(merged.hhp_index_gemini).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3)
        ax.set_xlabel("HHP-Index (OpenAI text-embedding-3-large)")
        ax.set_ylabel("HHP-Index (Gemini gemini-embedding-2-preview)")
        ax.set_title(f"Cross-Model HHP-Index Correlation\nr={r_pearson:.3f}, ρ={r_spearman:.3f}")
        ax.legend()
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        plt.tight_layout()
        plt.savefig(gemini_dir / "cross_model_correlation.png", dpi=150)
        plt.close()
        print(f"\n  cross_model_correlation.png")
    except Exception as e:
        print(f"  [W] {e}")

    # 69 triplet comparison
    print(f"\n  --- 69 Triplet Cross-Model ---")
    for dir_name, dir_path, label in [("OpenAI", openai_dir, "OAI"), ("Gemini", gemini_dir, "GEM")]:
        p3b = dir_path / "phase3b_number_triplet.csv"
        if p3b.exists():
            df_num = pd.read_csv(p3b)
            print(f"  [{label}]  " + "  ".join(f"{int(r['number'])}:{r['pull']:+.4f}" for _, r in df_num.iterrows()))

    print(f"\n{'='*70}")
    print("CROSS-VALIDATION COMPLETE")
    print(f"{'='*70}\n")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HHP Gemini Cross-Validation")
    sub = parser.add_subparsers(dest="command")

    # Gemini experiment
    run_parser = sub.add_parser("run", help="Run HHP experiment with Gemini")
    run_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)

    # Cross-model comparison
    cmp_parser = sub.add_parser("compare", help="Compare OpenAI vs Gemini results")
    cmp_parser.add_argument("--openai-dir", default="hhp_results_v3")
    cmp_parser.add_argument("--gemini-dir", default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()

    if args.command == "run":
        run_gemini_experiment(args)
    elif args.command == "compare":
        run_cross_compare(Path(args.openai_dir), Path(args.gemini_dir))
    else:
        # Default: run both
        print("Usage:")
        print("  python hhp_gemini_cross.py run      # Run Gemini experiment")
        print("  python hhp_gemini_cross.py compare   # Compare OpenAI vs Gemini")
