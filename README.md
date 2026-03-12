# HHP Experiment: Computational Detection of Covert Second-Path Meanings

## Quick Start

```bash
# 1. Install dependencies
pip install -r hhp_requirements.txt

# 2. Set API key
export OPENAI_API_KEY="sk-your-key-here"
# Or create .env file:
# OPENAI_API_KEY=sk-your-key-here

# 3. Run experiment
python hhp_experiment.py

# 4. Check results
ls hhp_results/
```

## Output Files

| File | Description |
|---|---|
| `raw_embeddings.json` | All embeddings (for re-analysis without re-calling API) |
| `phase1_word_level.csv` | Word-level HHP-Index scores |
| `phase2_context_separation.csv` | Context sensitivity scores |
| `phase3_reactivation.csv` | Cue-based reactivation scores |
| `phase1_hhp_index.png` | Bar chart: HHP-Index per word |
| `phase1_pca.png` | PCA: words in embedding space |
| `phase1_umap.png` | UMAP: words in embedding space |
| `phase2_context_sensitivity.png` | Context separation per word |
| `phase1_domain_analysis.png` | Mean HHP-Index by semantic domain |

## Experiment Design

### Phase 1: Word-Level Analysis
- Embed HHP candidates and controls as bare words
- Compute cosine similarity to sexual anchor cluster vs neutral anchor cluster
- HHP-Index = sim(word, sexual) - sim(word, neutral)
- **Prediction:** HHP candidates > Controls

### Phase 2: Context Template Analysis
- Embed same word in 3 contexts: primary meaning / second path / neutral
- Measure cosine distance between primary and second-path embeddings
- **Prediction:** HHP candidates show larger context separation

### Phase 3: Cue-Based Reactivation
- Embed HHP candidates in PRIMARY-meaning-only sentences
- Measure residual "pull" toward sexual cluster
- **Prediction:** Even in innocent context, HHP candidates leak toward sexual cluster

## Estimated API Cost
- ~300 embedding calls × text-embedding-3-large
- Estimated cost: < $0.50 USD

## Languages
English, Japanese, French, German, Spanish, Korean, Chinese
