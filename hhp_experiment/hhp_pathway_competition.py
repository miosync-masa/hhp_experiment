#!/usr/bin/env python3
"""
HHP Pathway Competition Analysis
==================================
Tests the hypothesis that "master" appears far from the sexual cluster
not because its second pathway is weak, but because a third pathway
(slavery/colonialism) masks it.

Cross-linguistic comparison:
  master (en) → slavery pathway competes with BDSM pathway
  ご主人様 (ja) → no slavery pathway → BDSM pathway unmasked
  Meister (de) → craft/guild pathway → neutral
  maître (fr)  → authority pathway → moderate

If confirmed: competing pathways explain why some HHP candidates
score low on HHP-Index despite having community-shared second paths.

This bridges HHP (lexical layer) with MBL (lens layer):
  Translation-equivalent words occupy different embedding positions
  because culture-specific competing pathways shift the vector.

Author: Masamichi Iizumi / Miosync, Inc.
Date: 2026-03-12

Usage:
    # Uses cached embeddings from v3 + optionally calls API for new terms
    python hhp_pathway_competition.py --cache hhp_results_v3/embedding_cache.json
    python hhp_pathway_competition.py --cache hhp_results_gemini/gemini_embedding_cache.json --model gemini
"""

from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Target Words: Translation Equivalents of "master"
# =============================================================================

MASTER_EQUIVALENTS = {
    "master_en":      {"word": "master",     "lang": "en", "label": "master (EN)"},
    "goshujinsama_ja":{"word": "ご主人様",    "lang": "ja", "label": "ご主人様 (JA)"},
    "meister_de":     {"word": "Meister",    "lang": "de", "label": "Meister (DE)"},
    "maitre_fr":      {"word": "maître",     "lang": "fr", "label": "maître (FR)"},
    "amo_es":         {"word": "amo",        "lang": "es", "label": "amo (ES)"},
    "juyin_ko":       {"word": "주인님",      "lang": "ko", "label": "주인님 (KO)"},
    "zhuren_zh":      {"word": "主人",        "lang": "zh", "label": "主人 (ZH)"},
}

# Additional HHP words with potential competing pathways
COMPETING_PATHWAY_WORDS = {
    "dominate_en":    {"word": "dominate",   "lang": "en", "label": "dominate (EN)"},
    "shihai_ja":      {"word": "支配",        "lang": "ja", "label": "支配 (JA)"},
    "submit_en":      {"word": "submit",     "lang": "en", "label": "submit (EN)"},
    "slave_en":       {"word": "slave",      "lang": "en", "label": "slave (EN)"},
    "dorei_ja":       {"word": "奴隷",        "lang": "ja", "label": "奴隷 (JA)"},
    "collar_en":      {"word": "collar",     "lang": "en", "label": "collar (EN)"},
    "kubiwa_ja":      {"word": "首輪",        "lang": "ja", "label": "首輪 (JA)"},
    "bind_en":        {"word": "bind",       "lang": "en", "label": "bind (EN)"},
    "shibaru_ja":     {"word": "縛る",        "lang": "ja", "label": "縛る (JA)"},
}

# =============================================================================
# Anchor Clusters (3 competing attractors)
# =============================================================================

SEXUAL_ANCHORS = [
    "sexual intercourse", "orgasm", "erotic", "BDSM", "bondage",
    "dominatrix", "submissive", "sadomasochism", "fetish", "kink",
    "性行為", "SM", "ボンデージ", "フェティシズム", "調教",
]

SLAVERY_ANCHORS = [
    "slavery", "slave trade", "plantation", "abolition", "colonialism",
    "oppression", "emancipation", "human trafficking", "forced labor",
    "racial oppression", "antebellum", "middle passage",
]

NEUTRAL_ANCHORS = [
    "mathematics", "engineering", "accounting", "weather forecast",
    "bus schedule", "office meeting", "calendar", "recipe",
    "数学", "会計", "天気予報", "会議",
]

CRAFT_ANCHORS = [
    "craftsman", "apprentice", "guild", "artisan", "workshop",
    "expertise", "mastery", "journeyman", "skill", "profession",
    "職人", "弟子", "修行", "匠", "技能",
]

# Context sentences for master-equivalents
CONTEXT_SENTENCES = {
    # Primary (craft/authority)
    "master_primary_en":   "The master craftsman trained his apprentice for years.",
    "master_primary_ja":   "ご主人様は職人として弟子を育てた。",
    "master_primary_de":   "Der Meister bildete seinen Lehrling sorgfältig aus.",
    "master_primary_fr":   "Le maître artisan a formé son apprenti pendant des années.",
    # Sexual (BDSM)
    "master_sexual_en":    "Call me master, she whispered submissively.",
    "master_sexual_ja":    "ご主人様と呼んで、と彼女は従順に囁いた。",
    "master_sexual_de":    "Nenn mich Meister, flüsterte sie unterwürfig.",
    "master_sexual_fr":    "Appelez-moi maître, murmura-t-elle soumise.",
    # Slavery (historical)
    "master_slavery_en":   "The master owned hundreds of slaves on his plantation.",
    "master_slavery_ja":   "主人はプランテーションで数百人の奴隷を所有していた。",
    "master_slavery_de":   "Der Herr besaß Hunderte von Sklaven auf seiner Plantage.",
    "master_slavery_fr":   "Le maître possédait des centaines d'esclaves dans sa plantation.",
}


# =============================================================================
# Embedding functions
# =============================================================================

def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("embeddings", {})


def get_openai_embedding(client, text, model="text-embedding-3-large"):
    try:
        resp = client.embeddings.create(input=text, model=model)
        return np.array(resp.data[0].embedding)
    except Exception as e:
        print(f"  [ERROR] '{text[:40]}': {e}")
        return None


def get_gemini_embedding(text, model="gemini-embedding-2-preview"):
    import google.generativeai as genai
    try:
        resp = genai.embed_content(model=f"models/{model}", content=text)
        return np.array(resp['embedding'])
    except Exception as e:
        print(f"  [ERROR] '{text[:40]}': {e}")
        return None


def embed_all(texts, cache, model_type="openai"):
    """Embed texts, using cache where possible."""
    result = {}
    pending = []

    for t in texts:
        if t in cache:
            result[t] = np.array(cache[t])
        else:
            pending.append(t)

    print(f"  Cached: {len(result)} / Pending: {len(pending)}")

    if pending:
        if model_type == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            for t in pending:
                emb = get_openai_embedding(client, t)
                if emb is not None:
                    result[t] = emb
                    cache[t] = emb.tolist()
                time.sleep(0.1)
        elif model_type == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            for t in pending:
                emb = get_gemini_embedding(t)
                if emb is not None:
                    result[t] = emb
                    cache[t] = emb.tolist()
                time.sleep(0.1)

    return result, cache


def l2_normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def sim(a, b):
    return float(cosine_similarity([a], [b])[0][0])


def centroid(embs, texts):
    vecs = [embs[t] for t in texts if t in embs]
    if not vecs: return None
    return l2_normalize(np.mean(vecs, axis=0))


# =============================================================================
# Main Analysis
# =============================================================================

def run_analysis(args):
    print("=" * 70)
    print("HHP PATHWAY COMPETITION ANALYSIS")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Time: {datetime.now().isoformat()}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all texts
    all_texts = set()
    for d in [MASTER_EQUIVALENTS, COMPETING_PATHWAY_WORDS]:
        for v in d.values():
            all_texts.add(v["word"])
    for anchors in [SEXUAL_ANCHORS, SLAVERY_ANCHORS, NEUTRAL_ANCHORS, CRAFT_ANCHORS]:
        all_texts.update(anchors)
    all_texts.update(CONTEXT_SENTENCES.values())
    all_texts = list(all_texts)
    print(f"Total texts: {len(all_texts)}")

    # Load cache and embed
    cache = load_cache(Path(args.cache))
    embs, cache = embed_all(all_texts, cache, model_type=args.model)
    print(f"Embedded: {len(embs)}/{len(all_texts)}")

    # Save updated cache
    cache_out = output_dir / f"pathway_cache_{args.model}.json"
    with open(cache_out, "w", encoding="utf-8") as f:
        json.dump({"embeddings": {k: v.tolist() if isinstance(v, np.ndarray) else v
                                   for k, v in cache.items()}}, f, ensure_ascii=False)

    # Normalize
    embs = {k: l2_normalize(v) for k, v in embs.items()}

    # Compute centroids
    sex_c = centroid(embs, SEXUAL_ANCHORS)
    slave_c = centroid(embs, SLAVERY_ANCHORS)
    neut_c = centroid(embs, NEUTRAL_ANCHORS)
    craft_c = centroid(embs, CRAFT_ANCHORS)

    if any(c is None for c in [sex_c, slave_c, neut_c, craft_c]):
        print("[FATAL] Could not compute all centroids")
        return

    # =================================================================
    # Analysis 1: Master-equivalents attraction map
    # =================================================================
    print("\n" + "=" * 70)
    print("[Analysis 1] MASTER-EQUIVALENTS: FOUR-WAY ATTRACTION")
    print("=" * 70)

    print(f"\n  {'Word':20s} | {'→Sexual':>8s} | {'→Slavery':>9s} | {'→Craft':>8s} | {'→Neutral':>9s} | {'Sex-Slav':>9s}")
    print("  " + "-" * 78)

    rows = []
    for key, info in MASTER_EQUIVALENTS.items():
        w = info["word"]
        if w not in embs:
            continue
        e = embs[w]
        s_sex = sim(e, sex_c)
        s_slav = sim(e, slave_c)
        s_craft = sim(e, craft_c)
        s_neut = sim(e, neut_c)
        delta = s_sex - s_slav  # positive = sexual wins over slavery

        row = {"key": key, "word": w, "lang": info["lang"], "label": info["label"],
               "sim_sexual": s_sex, "sim_slavery": s_slav,
               "sim_craft": s_craft, "sim_neutral": s_neut,
               "sex_minus_slavery": delta}
        rows.append(row)

        marker = " ✓" if delta > 0 else ""
        print(f"  {info['label']:20s} | {s_sex:8.4f} | {s_slav:9.4f} | {s_craft:8.4f} | {s_neut:9.4f} | {delta:+9.4f}{marker}")

    df_master = pd.DataFrame(rows)
    df_master.to_csv(output_dir / "master_equivalents.csv", index=False, encoding="utf-8-sig")

    # =================================================================
    # Analysis 2: Context-dependent attraction
    # =================================================================
    print("\n" + "=" * 70)
    print("[Analysis 2] CONTEXT SENTENCES: WHERE DOES EACH PULL?")
    print("=" * 70)

    ctx_rows = []
    print(f"\n  {'Context':40s} | {'→Sexual':>8s} | {'→Slavery':>9s} | {'→Craft':>8s} | {'Winner':>10s}")
    print("  " + "-" * 82)

    for key, sent in CONTEXT_SENTENCES.items():
        if sent not in embs:
            continue
        e = embs[sent]
        s_sex = sim(e, sex_c)
        s_slav = sim(e, slave_c)
        s_craft = sim(e, craft_c)

        scores = {"sexual": s_sex, "slavery": s_slav, "craft": s_craft}
        winner = max(scores, key=scores.get)

        ctx_rows.append({"key": key, "sentence": sent[:60],
                        "sim_sexual": s_sex, "sim_slavery": s_slav, "sim_craft": s_craft,
                        "winner": winner})
        print(f"  {key:40s} | {s_sex:8.4f} | {s_slav:9.4f} | {s_craft:8.4f} | {winner:>10s}")

    df_ctx = pd.DataFrame(ctx_rows)
    df_ctx.to_csv(output_dir / "context_attraction.csv", index=False, encoding="utf-8-sig")

    # =================================================================
    # Analysis 3: EN vs JA direct comparison
    # =================================================================
    print("\n" + "=" * 70)
    print("[Analysis 3] ENGLISH vs JAPANESE PATHWAY COMPETITION")
    print("=" * 70)

    pairs = [
        ("master", "ご主人様", "master/authority"),
        ("dominate", "支配", "power/control"),
        ("slave", "奴隷", "slavery/servitude"),
        ("collar", "首輪", "restraint"),
        ("bind", "縛る", "binding"),
        ("submit", "submit_only", "submission"),
    ]

    print(f"\n  {'Concept':15s} | {'EN→Sex':>8s} {'EN→Slav':>8s} {'EN Δ':>8s} | {'JA→Sex':>8s} {'JA→Slav':>8s} {'JA Δ':>8s} | {'JA-EN Δ':>8s}")
    print("  " + "-" * 95)

    comparison_rows = []
    for en_word, ja_word, concept in pairs:
        if en_word not in embs or ja_word not in embs:
            continue
        en_sex = sim(embs[en_word], sex_c)
        en_slav = sim(embs[en_word], slave_c)
        en_delta = en_sex - en_slav

        ja_sex = sim(embs[ja_word], sex_c)
        ja_slav = sim(embs[ja_word], slave_c)
        ja_delta = ja_sex - ja_slav

        cross_delta = ja_delta - en_delta

        comparison_rows.append({
            "concept": concept, "en_word": en_word, "ja_word": ja_word,
            "en_sim_sexual": en_sex, "en_sim_slavery": en_slav, "en_delta": en_delta,
            "ja_sim_sexual": ja_sex, "ja_sim_slavery": ja_slav, "ja_delta": ja_delta,
            "cross_delta": cross_delta,
        })

        marker = " ✓" if cross_delta > 0 else ""
        print(f"  {concept:15s} | {en_sex:8.4f} {en_slav:8.4f} {en_delta:+8.4f} | {ja_sex:8.4f} {ja_slav:8.4f} {ja_delta:+8.4f} | {cross_delta:+8.4f}{marker}")

    if comparison_rows:
        df_comp = pd.DataFrame(comparison_rows)
        df_comp.to_csv(output_dir / "en_ja_comparison.csv", index=False, encoding="utf-8-sig")

        deltas = df_comp["cross_delta"].values
        print(f"\n  Mean cross-delta (JA-EN): {np.mean(deltas):+.4f}")
        print(f"  JA more sexual: {np.sum(deltas > 0)}/{len(deltas)}")

    # =================================================================
    # Analysis 4: Additional competing pathway words
    # =================================================================
    print("\n" + "=" * 70)
    print("[Analysis 4] ADDITIONAL WORDS: THREE-WAY ATTRACTION")
    print("=" * 70)

    print(f"\n  {'Word':20s} | {'→Sexual':>8s} | {'→Slavery':>9s} | {'→Craft':>8s} | {'Dominant':>10s}")
    print("  " + "-" * 66)

    for key, info in COMPETING_PATHWAY_WORDS.items():
        w = info["word"]
        if w not in embs: continue
        e = embs[w]
        s_sex = sim(e, sex_c)
        s_slav = sim(e, slave_c)
        s_craft = sim(e, craft_c)
        scores = {"sexual": s_sex, "slavery": s_slav, "craft": s_craft}
        winner = max(scores, key=scores.get)
        print(f"  {info['label']:20s} | {s_sex:8.4f} | {s_slav:9.4f} | {s_craft:8.4f} | {winner:>10s}")

    # =================================================================
    # PLOTS
    # =================================================================
    print("\n" + "=" * 70)
    print("[Plots]")
    print("=" * 70)

    # Plot 1: Master-equivalents radar/bar
    try:
        if len(df_master) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Left: Grouped bar chart
            labels = df_master["label"].values
            x = np.arange(len(labels))
            w = 0.2
            axes[0].bar(x - 1.5*w, df_master["sim_sexual"], w, label="→ Sexual", color="#e74c3c", alpha=0.8)
            axes[0].bar(x - 0.5*w, df_master["sim_slavery"], w, label="→ Slavery", color="#2c3e50", alpha=0.8)
            axes[0].bar(x + 0.5*w, df_master["sim_craft"], w, label="→ Craft", color="#f39c12", alpha=0.8)
            axes[0].bar(x + 1.5*w, df_master["sim_neutral"], w, label="→ Neutral", color="#95a5a6", alpha=0.8)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
            axes[0].set_ylabel("Cosine Similarity")
            axes[0].set_title(f"Four-Way Attraction Map ({args.model})")
            axes[0].legend(fontsize=8)

            # Right: Sex-Slavery delta
            colors = ["#e74c3c" if d > 0 else "#2c3e50" for d in df_master["sex_minus_slavery"]]
            axes[1].barh(range(len(df_master)), df_master["sex_minus_slavery"], color=colors)
            axes[1].set_yticks(range(len(df_master)))
            axes[1].set_yticklabels(df_master["label"], fontsize=9)
            axes[1].set_xlabel("Sexual − Slavery (positive = sexual wins)")
            axes[1].set_title("Pathway Competition: Sexual vs Slavery")
            axes[1].axvline(0, color="gray", ls="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(output_dir / "master_attraction_map.png", dpi=150)
            plt.close()
            print("  master_attraction_map.png")
    except Exception as e:
        print(f"  [W] {e}")

    # Plot 2: EN vs JA comparison
    try:
        if comparison_rows:
            fig, ax = plt.subplots(figsize=(10, 6))
            concepts = [r["concept"] for r in comparison_rows]
            en_deltas = [r["en_delta"] for r in comparison_rows]
            ja_deltas = [r["ja_delta"] for r in comparison_rows]

            x = np.arange(len(concepts))
            ax.bar(x - 0.15, en_deltas, 0.3, label="English (Sexual−Slavery)", color="#3498db", alpha=0.8)
            ax.bar(x + 0.15, ja_deltas, 0.3, label="Japanese (Sexual−Slavery)", color="#e74c3c", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(concepts, rotation=30, ha='right')
            ax.set_ylabel("Sexual − Slavery delta")
            ax.set_title("Cross-Linguistic Pathway Competition\n(Positive = sexual pathway dominates)")
            ax.axhline(0, color="gray", ls="--", alpha=0.5)
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "en_ja_pathway_competition.png", dpi=150)
            plt.close()
            print("  en_ja_pathway_competition.png")
    except Exception as e:
        print(f"  [W] {e}")

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("PATHWAY COMPETITION ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nKey findings:")
    print(f"  1. If ご主人様 is closer to sexual cluster than 'master',")
    print(f"     slavery pathway is masking master's sexual second path")
    print(f"  2. If cross-delta (JA-EN) is positive for most pairs,")
    print(f"     Japanese lacks the competing slavery pathway")
    print(f"  3. Embedding position difference between translation-equivalents")
    print(f"     = cultural lens difference (MBL connection)")
    print()

    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name:40s} {f.stat().st_size:>10,} bytes")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HHP Pathway Competition Analysis")
    parser.add_argument("--cache", default="hhp_results_v3/embedding_cache.json",
                       help="Path to embedding cache file")
    parser.add_argument("--model", default="openai", choices=["openai", "gemini"],
                       help="Which model to use for new embeddings")
    parser.add_argument("--output-dir", default="hhp_results_pathway",
                       help="Output directory")
    args = parser.parse_args()
    run_analysis(args)
