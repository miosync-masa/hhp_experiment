#!/usr/bin/env python3
"""
HHP (Homographic Homophonic Polysemy) Experiment v3
===================================================
Computational detection of covert second-path meanings in embedding space.

Key changes from v2
-------------------
1. True batch embedding requests with persistent on-disk cache
2. Language-aware anchor centroids (fallback to global)
3. Safer statistics: Mann-Whitney / Welch t / permutation / bootstrap CI / Hedges g
4. Cleaner phase structure:
   - Phase 1: word-level HHP index
   - Phase 2: context separation (HHP only) + leakage from primary context (HHP vs Control)
   - Phase 3: matched-pair primary-context leakage
   - Phase 3b: 69 number triplet analysis
5. Stronger reproducibility: config dump, versioned outputs, seed control, resume support
6. More defensible controls: no eroticized "second sense" sentences for control items
7. Automatic language-specific centroids to reduce cross-lingual anchor bleed

Author: Masamichi Iizumi / Miosync, Inc.
Date: 2026-03-12

Requirements:
    pip install openai numpy pandas matplotlib scikit-learn scipy python-dotenv

Usage:
    export OPENAI_API_KEY="sk-..."
    python hhp_experiment_v3.py --output-dir hhp_results_v3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import mannwhitneyu, ttest_ind, ttest_1samp, wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_OUTPUT_DIR = "hhp_results_v3"
DEFAULT_CACHE_FILE = "embedding_cache.json"
DEFAULT_BATCH_SIZE = 128
DEFAULT_API_DELAY = 0.0
DEFAULT_RANDOM_SEED = 42
DEFAULT_BOOTSTRAP_ITERS = 5000
DEFAULT_PERMUTATION_ITERS = 10000

LANGS = ["en", "ja", "fr", "de", "es", "ko", "zh"]


# =============================================================================
# Data inventory
# =============================================================================

HHP_CANDIDATES = {
    "come_en":       {"word": "come",      "lang": "en", "domain": "movement",    "primary": "to move toward",           "second_path": "orgasm"},
    "ride_en":       {"word": "ride",      "lang": "en", "domain": "movement",    "primary": "to sit on and travel",     "second_path": "sexual position"},
    "hot_en":        {"word": "hot",       "lang": "en", "domain": "temperature", "primary": "high temperature",          "second_path": "sexually attractive"},
    "wet_en":        {"word": "wet",       "lang": "en", "domain": "moisture",    "primary": "covered in water",          "second_path": "sexual arousal"},
    "blow_en":       {"word": "blow",      "lang": "en", "domain": "air",         "primary": "to push air",               "second_path": "oral sex"},
    "eat_en":        {"word": "eat",       "lang": "en", "domain": "food",        "primary": "to consume food",           "second_path": "oral sex"},
    "mount_en":      {"word": "mount",     "lang": "en", "domain": "movement",    "primary": "to climb on top",           "second_path": "sexual mounting"},
    "lay_en":        {"word": "lay",       "lang": "en", "domain": "position",    "primary": "to put down flat",          "second_path": "to have sex with"},
    "master_en":     {"word": "master",    "lang": "en", "domain": "authority",   "primary": "a person in control",       "second_path": "BDSM dominant"},
    "69_en":         {"word": "69",        "lang": "en", "domain": "number",      "primary": "a number",                  "second_path": "sexual position"},
    "doggy_en":      {"word": "doggy",     "lang": "en", "domain": "animal",      "primary": "resembling a dog",          "second_path": "sexual position"},
    "nuts_en":       {"word": "nuts",      "lang": "en", "domain": "food",        "primary": "tree seeds",                "second_path": "testicles"},
    "screw_en":      {"word": "screw",     "lang": "en", "domain": "tool",        "primary": "a metal fastener",          "second_path": "to have sex"},
    "collar_en":     {"word": "collar",    "lang": "en", "domain": "animal",      "primary": "band around neck",          "second_path": "BDSM collar"},
    "leash_en":      {"word": "leash",     "lang": "en", "domain": "animal",      "primary": "strap for animal",          "second_path": "BDSM restraint"},
    "dominate_en":   {"word": "dominate",  "lang": "en", "domain": "power",       "primary": "to have control",           "second_path": "BDSM domination"},
    "submit_en":     {"word": "submit",    "lang": "en", "domain": "power",       "primary": "to yield",                  "second_path": "sexual submission"},
    "swallow_en":    {"word": "swallow",   "lang": "en", "domain": "food",        "primary": "to pass food down throat",  "second_path": "oral sex act"},
    "iku_ja":        {"word": "イク",       "lang": "ja", "domain": "movement",    "primary": "to go",                     "second_path": "orgasm"},
    "taberu_ja":     {"word": "食べる",     "lang": "ja", "domain": "food",        "primary": "to eat",                    "second_path": "desire expression"},
    "noru_ja":       {"word": "乗る",       "lang": "ja", "domain": "movement",    "primary": "to ride",                   "second_path": "sexual position"},
    "kubiwa_ja":     {"word": "首輪",       "lang": "ja", "domain": "animal",      "primary": "pet collar",                "second_path": "BDSM collar"},
    "nureru_ja":     {"word": "濡れる",     "lang": "ja", "domain": "moisture",    "primary": "to get wet",                "second_path": "sexual arousal"},
    "semeru_ja":     {"word": "攻める",     "lang": "ja", "domain": "power",       "primary": "to attack",                 "second_path": "sexual aggression"},
    "shibaru_ja":    {"word": "縛る",       "lang": "ja", "domain": "tool",        "primary": "to tie/bind",               "second_path": "bondage"},
    "neru_ja":       {"word": "寝る",       "lang": "ja", "domain": "sleep",       "primary": "to sleep",                  "second_path": "to have sex"},
    "venir_fr":      {"word": "venir",     "lang": "fr", "domain": "movement",    "primary": "to come",                   "second_path": "orgasm"},
    "mourir_fr":     {"word": "mourir",    "lang": "fr", "domain": "state",       "primary": "to die",                    "second_path": "orgasm (la petite mort)"},
    "monter_fr":     {"word": "monter",    "lang": "fr", "domain": "movement",    "primary": "to go up",                  "second_path": "sexual mounting"},
    "kommen_de":     {"word": "kommen",    "lang": "de", "domain": "movement",    "primary": "to come",                   "second_path": "orgasm"},
    "reiten_de":     {"word": "reiten",    "lang": "de", "domain": "movement",    "primary": "to ride (horse)",           "second_path": "sexual position"},
    "blasen_de":     {"word": "blasen",    "lang": "de", "domain": "air",         "primary": "to blow",                   "second_path": "oral sex"},
    "correrse_es":   {"word": "correrse",  "lang": "es", "domain": "movement",    "primary": "to run/slide",              "second_path": "orgasm"},
    "montar_es":     {"word": "montar",    "lang": "es", "domain": "movement",    "primary": "to mount/ride",             "second_path": "to have sex"},
    "gada_ko":       {"word": "가다",       "lang": "ko", "domain": "movement",    "primary": "to go",                     "second_path": "orgasm"},
    "lai_zh":        {"word": "来",         "lang": "zh", "domain": "movement",    "primary": "to come",                   "second_path": "orgasm"},
    # ---------- English ----------
    "lick_en":       {"word": "lick",      "lang": "en", "domain": "food",        "primary": "to pass tongue over",         "second_path": "oral sex"},
    "tie_en":        {"word": "tie",       "lang": "en", "domain": "tool",        "primary": "to fasten with cord",         "second_path": "bondage"},
    "obey_en":       {"word": "obey",      "lang": "en", "domain": "power",       "primary": "to follow orders",            "second_path": "sexual submission"},
    # ---------- Japanese ----------
    "nameru_ja":     {"word": "舐める",     "lang": "ja", "domain": "food",        "primary": "to lick",                     "second_path": "oral sex"},
    "shitagau_ja":   {"word": "従う",       "lang": "ja", "domain": "power",       "primary": "to obey/follow",              "second_path": "sexual submission"},
    # ---------- French ----------
    "chaud_fr":      {"word": "chaud",     "lang": "fr", "domain": "temperature", "primary": "hot/warm",                    "second_path": "sexually attractive/aroused"},
    "mouiller_fr":   {"word": "mouiller",  "lang": "fr", "domain": "moisture",    "primary": "to wet",                      "second_path": "sexual arousal"},
    # ---------- German ----------
    "heiss_de":      {"word": "heiß",      "lang": "de", "domain": "temperature", "primary": "hot",                         "second_path": "sexually attractive/aroused"},
    "feucht_de":     {"word": "feucht",    "lang": "de", "domain": "moisture",    "primary": "moist",                       "second_path": "sexual arousal"},

    # ---------- Spanish ----------
    "caliente_es":   {"word": "caliente",  "lang": "es", "domain": "temperature", "primary": "hot",                         "second_path": "sexually attractive/aroused"},
    "mojarse_es":    {"word": "mojarse",   "lang": "es", "domain": "moisture",    "primary": "to get wet",                  "second_path": "sexual arousal"},  
}

CONTROL_WORDS = {
    "walk_en":       {"word": "walk",      "lang": "en", "domain": "movement",    "primary": "to move on foot"},
    "sit_en":        {"word": "sit",       "lang": "en", "domain": "position",    "primary": "to be seated"},
    "cold_en":       {"word": "cold",      "lang": "en", "domain": "temperature", "primary": "low temperature"},
    "dry_en":        {"word": "dry",       "lang": "en", "domain": "moisture",    "primary": "free from moisture"},
    "cup_en":        {"word": "cup",       "lang": "en", "domain": "tool",        "primary": "drinking vessel"},
    "desk_en":       {"word": "desk",      "lang": "en", "domain": "tool",        "primary": "a table for work"},
    "pencil_en":     {"word": "pencil",    "lang": "en", "domain": "tool",        "primary": "writing instrument"},
    "68_en":         {"word": "68",        "lang": "en", "domain": "number",      "primary": "a number"},
    "70_en":         {"word": "70",        "lang": "en", "domain": "number",      "primary": "a number"},
    "breathe_en":    {"word": "breathe",   "lang": "en", "domain": "air",         "primary": "to inhale and exhale"},
    "carry_en":      {"word": "carry",     "lang": "en", "domain": "movement",    "primary": "to transport"},
    "table_en":      {"word": "table",     "lang": "en", "domain": "tool",        "primary": "a piece of furniture"},
    "aruku_ja":      {"word": "歩く",       "lang": "ja", "domain": "movement",    "primary": "to walk"},
    "suwaru_ja":     {"word": "座る",       "lang": "ja", "domain": "position",    "primary": "to sit"},
    "koppu_ja":      {"word": "コップ",     "lang": "ja", "domain": "tool",        "primary": "cup"},
    "tsukue_ja":     {"word": "机",         "lang": "ja", "domain": "tool",        "primary": "desk"},
    "enpitsu_ja":    {"word": "鉛筆",       "lang": "ja", "domain": "tool",        "primary": "pencil"},
    "marcher_fr":    {"word": "marcher",   "lang": "fr", "domain": "movement",    "primary": "to walk"},
    "asseoir_fr":    {"word": "s'asseoir", "lang": "fr", "domain": "position",    "primary": "to sit"},
    "gehen_de":      {"word": "gehen",     "lang": "de", "domain": "movement",    "primary": "to walk/go"},
    "sitzen_de":     {"word": "sitzen",    "lang": "de", "domain": "position",    "primary": "to sit"},
    "geotda_ko":     {"word": "걷다",       "lang": "ko", "domain": "movement",    "primary": "to walk"},
    "zou_zh":        {"word": "走",         "lang": "zh", "domain": "movement",    "primary": "to walk"},
    # ---------- English ----------
    "chew_en":       {"word": "chew",       "lang": "en", "domain": "food",        "primary": "to bite and grind food"},
    "fasten_en":     {"word": "fasten",     "lang": "en", "domain": "tool",        "primary": "to secure or attach"},
    "manage_en":     {"word": "manage",     "lang": "en", "domain": "power",       "primary": "to administer or control"},
    "warm_en":       {"word": "warm",       "lang": "en", "domain": "temperature", "primary": "moderately hot"},
    "damp_en":       {"word": "damp",       "lang": "en", "domain": "moisture",    "primary": "slightly wet"},
    # ---------- Japanese ----------
    "kamu_ja":       {"word": "噛む",        "lang": "ja", "domain": "food",        "primary": "to chew"},
    "koteisuru_ja":  {"word": "固定する",    "lang": "ja", "domain": "tool",        "primary": "to fix/secure"},
    "atarakai_ja":   {"word": "温かい",      "lang": "ja", "domain": "temperature", "primary": "warm"},
    "shimeru_ja":    {"word": "湿る",        "lang": "ja", "domain": "moisture",    "primary": "to become damp"},
    # ---------- French ----------
    "tiede_fr":      {"word": "tiède",      "lang": "fr", "domain": "temperature", "primary": "lukewarm"},
    "humide_fr":     {"word": "humide",     "lang": "fr", "domain": "moisture",    "primary": "damp/moist"},
    # ---------- German ----------
    "kuhl_de":       {"word": "kühl",       "lang": "de", "domain": "temperature", "primary": "cool"},
    "trocken_de":    {"word": "trocken",    "lang": "de", "domain": "moisture",    "primary": "dry"},
    # ---------- Spanish ----------
    "templado_es":   {"word": "templado",   "lang": "es", "domain": "temperature", "primary": "mild/warm"},
    "seco_es":       {"word": "seco",       "lang": "es", "domain": "moisture",    "primary": "dry"},
}

ANCHORS = {
    "sexual": {
        "en": ["sexual intercourse", "orgasm", "erotic", "libido", "arousal", "foreplay", "intimacy", "seduction", "sensual pleasure"],
        "ja": ["性行為", "オーガズム", "エロティック", "性的興奮", "快感", "前戯", "親密さ"],
        "fr": ["rapport sexuel", "orgasme", "érotique", "excitation sexuelle"],
        "de": ["Geschlechtsverkehr", "Orgasmus", "erotisch", "sexuelle Erregung"],
        "es": ["relación sexual", "orgasmo", "erótico", "excitación sexual"],
        "ko": ["성행위", "오르가즘", "에로틱", "성적 흥분"],
        "zh": ["性行为", "高潮", "色情", "性兴奋"],
    },
    "neutral": {
        "en": ["mathematics", "accounting", "engineering", "weather forecast", "bus schedule", "office meeting", "calendar", "recipe"],
        "ja": ["数学", "会計", "工学", "天気予報", "バスの時刻表", "会議", "予定表"],
        "fr": ["mathématiques", "comptabilité", "ingénierie", "météo", "réunion de bureau"],
        "de": ["Mathematik", "Buchhaltung", "Ingenieurwesen", "Wetterbericht", "Bürobesprechung"],
        "es": ["matemáticas", "contabilidad", "ingeniería", "pronóstico del tiempo", "reunión de oficina"],
        "ko": ["수학", "회계", "공학", "일기예보", "회의"],
        "zh": ["数学", "会计", "工程", "天气预报", "会议"],
    },
    "primary": {
        "movement": {
            "en": ["walking", "traveling", "commuting", "jogging"],
            "ja": ["歩行", "移動", "通勤", "移動する"],
            "fr": ["marcher", "voyager", "se déplacer"],
            "de": ["gehen", "reisen", "sich bewegen"],
            "es": ["caminar", "viajar", "moverse"],
            "ko": ["걷기", "이동", "통근"],
            "zh": ["步行", "移动", "通勤"],
        },
        "food": {
            "en": ["cooking", "recipe", "restaurant", "nutrition"],
            "ja": ["料理", "レストラン", "栄養", "食事"],
            "fr": ["cuisine", "restaurant", "nutrition"],
            "de": ["Kochen", "Restaurant", "Ernährung"],
            "es": ["cocina", "restaurante", "nutrición"],
            "ko": ["요리", "식당", "영양"],
            "zh": ["烹饪", "餐馆", "营养"],
        },
        "temperature": {
            "en": ["thermometer", "weather", "climate", "forecast"],
            "ja": ["温度計", "天気", "気候", "予報"],
            "fr": ["thermomètre", "météo", "climat"],
            "de": ["Thermometer", "Wetter", "Klima"],
            "es": ["termómetro", "clima", "tiempo"],
            "ko": ["온도계", "날씨", "기후"],
            "zh": ["温度计", "天气", "气候"],
        },
        "moisture": {
            "en": ["rain", "umbrella", "humidity", "towel"],
            "ja": ["雨", "傘", "湿度", "タオル"],
            "fr": ["pluie", "parapluie", "humidité"],
            "de": ["Regen", "Regenschirm", "Feuchtigkeit"],
            "es": ["lluvia", "paraguas", "humedad"],
            "ko": ["비", "우산", "습도"],
            "zh": ["雨", "雨伞", "湿度"],
        },
        "air": {
            "en": ["wind", "breeze", "ventilation", "fan"],
            "ja": ["風", "換気", "扇風機", "そよ風"],
            "fr": ["vent", "brise", "ventilation"],
            "de": ["Wind", "Brise", "Lüftung"],
            "es": ["viento", "brisa", "ventilación"],
            "ko": ["바람", "환기", "선풍기"],
            "zh": ["风", "通风", "风扇"],
        },
        "animal": {
            "en": ["veterinarian", "pet shop", "dog walking", "animal shelter"],
            "ja": ["獣医", "ペットショップ", "犬の散歩", "保護施設"],
            "fr": ["vétérinaire", "animalerie", "chien"],
            "de": ["Tierarzt", "Zoohandlung", "Hund"],
            "es": ["veterinario", "tienda de mascotas", "perro"],
            "ko": ["수의사", "애완동물 가게", "개"],
            "zh": ["兽医", "宠物店", "狗"],
        },
        "power": {
            "en": ["management", "leadership", "authority", "governance"],
            "ja": ["経営", "リーダーシップ", "権威", "統治"],
            "fr": ["gestion", "leadership", "autorité"],
            "de": ["Management", "Führung", "Autorität"],
            "es": ["gestión", "liderazgo", "autoridad"],
            "ko": ["경영", "리더십", "권위"],
            "zh": ["管理", "领导力", "权威"],
        },
        "number": {
            "en": ["arithmetic", "counting", "calculator", "mathematics"],
            "ja": ["算数", "計算", "電卓", "数学"],
            "fr": ["arithmétique", "compter", "calculatrice"],
            "de": ["Arithmetik", "zählen", "Taschenrechner"],
            "es": ["aritmética", "contar", "calculadora"],
            "ko": ["산수", "계산", "계산기"],
            "zh": ["算术", "计数", "计算器"],
        },
        "position": {
            "en": ["furniture", "chair", "posture", "ergonomics"],
            "ja": ["家具", "椅子", "姿勢", "人間工学"],
            "fr": ["meuble", "chaise", "posture"],
            "de": ["Möbel", "Stuhl", "Haltung"],
            "es": ["mueble", "silla", "postura"],
            "ko": ["가구", "의자", "자세"],
            "zh": ["家具", "椅子", "姿势"],
        },
        "tool": {
            "en": ["hardware", "workshop", "repair", "construction"],
            "ja": ["工具", "修理", "建設", "作業場"],
            "fr": ["outil", "réparation", "atelier"],
            "de": ["Werkzeug", "Reparatur", "Werkstatt"],
            "es": ["herramienta", "reparación", "taller"],
            "ko": ["도구", "수리", "작업장"],
            "zh": ["工具", "修理", "车间"],
        },
        "sleep": {
            "en": ["bedtime", "pillow", "alarm clock", "insomnia"],
            "ja": ["就寝", "枕", "目覚まし", "不眠"],
            "fr": ["sommeil", "oreiller", "réveil"],
            "de": ["Schlaf", "Kissen", "Wecker"],
            "es": ["sueño", "almohada", "despertador"],
            "ko": ["수면", "베개", "알람시계"],
            "zh": ["睡眠", "枕头", "闹钟"],
        },
        "state": {
            "en": ["existence", "mortality", "funeral", "memorial"],
            "ja": ["存在", "死亡率", "葬儀", "追悼"],
            "fr": ["existence", "mortalité", "funérailles"],
            "de": ["Existenz", "Sterblichkeit", "Beerdigung"],
            "es": ["existencia", "mortalidad", "funeral"],
            "ko": ["존재", "사망률", "장례"],
            "zh": ["存在", "死亡率", "葬礼"],
        },
        "authority": {
            "en": ["leadership", "control", "manager", "authority"],
            "ja": ["権威", "支配", "管理者", "統率"],
            "fr": ["autorité", "contrôle", "chef"],
            "de": ["Autorität", "Kontrolle", "Chef"],
            "es": ["autoridad", "control", "jefe"],
            "ko": ["권위", "통제", "관리자"],
            "zh": ["权威", "控制", "管理者"],
        },
    },
}


# =============================================================================
# Context templates
# =============================================================================


def build_context_templates() -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """
    Returns:
        hhp_templates: key -> {primary, second, neutral}
        control_templates: key -> {primary, alternate, neutral}
        matched_pairs: pair_key -> pair specification
    """
    hhp_templates = {
        "come_en":     {"primary": "Please come to the office tomorrow morning.",
                          "second":  "Oh god, I am going to come.",
                          "neutral": "The word 'come' appears in many sentences."},
        "ride_en":     {"primary": "I ride the bus to work every day.",
                          "second":  "She wants to ride him all night.",
                          "neutral": "The word 'ride' has multiple uses."},
        "hot_en":      {"primary": "The soup is too hot to eat right now.",
                          "second":  "She looked incredibly hot in that dress.",
                          "neutral": "The word 'hot' is commonly used."},
        "wet_en":      {"primary": "My shoes got wet in the rain.",
                          "second":  "She was already wet with anticipation.",
                          "neutral": "The word 'wet' describes moisture."},
        "blow_en":     {"primary": "The wind will blow from the north today.",
                          "second":  "She offered to blow him in the car.",
                          "neutral": "The word 'blow' relates to air movement."},
        "eat_en":      {"primary": "Let us eat lunch at the cafeteria.",
                          "second":  "He wanted to eat her out all night.",
                          "neutral": "The word 'eat' means to consume food."},
        "mount_en":    {"primary": "The climber began to mount the steep rock face.",
                          "second":  "He mounted her from behind aggressively.",
                          "neutral": "The word 'mount' means to climb on top."},
        "lay_en":      {"primary": "Please lay the books on the table.",
                          "second":  "He wanted to lay her that night.",
                          "neutral": "The word 'lay' means to put something down."},
        "master_en":   {"primary": "The master craftsman trained the apprentice.",
                          "second":  "Call me master, she whispered submissively.",
                          "neutral": "The word 'master' denotes authority."},
        "69_en":       {"primary": "The answer to question 5 is 69.",
                          "second":  "They got into a 69 position on the bed.",
                          "neutral": "The number 69 is between 68 and 70."},
        "doggy_en":    {"primary": "The doggy wagged its tail happily at the park.",
                          "second":  "They did it doggy style on the floor.",
                          "neutral": "The word 'doggy' relates to dogs."},
        "collar_en":   {"primary": "The dog's collar has its name tag attached.",
                          "second":  "He fastened the leather collar around her neck as she knelt.",
                          "neutral": "A collar is a band worn around the neck."},
        "leash_en":    {"primary": "Keep your dog on a leash in the park.",
                          "second":  "She handed him the leash and knelt at his feet.",
                          "neutral": "A leash is used to restrain animals."},
        "dominate_en": {"primary": "The company dominates the global smartphone market.",
                          "second":  "He loved to dominate her in the bedroom.",
                          "neutral": "To dominate means to have control over."},
        "submit_en":   {"primary": "Please submit your report by Friday.",
                          "second":  "She loved to submit to his every command.",
                          "neutral": "To submit means to yield or hand over."},
        "nuts_en":     {"primary": "I bought mixed nuts at the grocery store.",
                          "second":  "She grabbed his nuts and squeezed firmly.",
                          "neutral": "Nuts are seeds from trees."},
        "screw_en":    {"primary": "Hand me the screw and the screwdriver.",
                          "second":  "They went upstairs to screw all afternoon.",
                          "neutral": "A screw is a type of fastener."},
        "swallow_en":  {"primary": "Swallow the medicine with a glass of water.",
                          "second":  "She decided to swallow instead of spit.",
                          "neutral": "To swallow means to pass down the throat."},
        "iku_ja":      {"primary": "明日、学校に行くつもりです。",
                          "second":  "もうイク、イっちゃう。",
                          "neutral": "『イク』は移動を意味する動詞です。"},
        "taberu_ja":   {"primary": "お昼にカレーを食べるつもりです。",
                          "second":  "可愛すぎて食べちゃいたい。",
                          "neutral": "『食べる』は摂食を意味する動詞です。"},
        "noru_ja":     {"primary": "毎朝バスに乗って通勤しています。",
                          "second":  "彼女が上に乗って腰を動かした。",
                          "neutral": "『乗る』は乗り物に乗ることを意味します。"},
        "kubiwa_ja":   {"primary": "犬に新しい首輪を買ってあげた。",
                          "second":  "革の首輪をつけて跪かせた。",
                          "neutral": "首輪は動物の首につける輪です。"},
        "nureru_ja":   {"primary": "雨に濡れて服がびしょびしょだ。",
                          "second":  "もう濡れてるの、と彼が囁いた。",
                          "neutral": "『濡れる』は水分が付着することです。"},
        "semeru_ja":   {"primary": "敵の陣地を攻めるべきだ。",
                          "second":  "彼に激しく攻められて声が出た。",
                          "neutral": "『攻める』は攻撃することを意味します。"},
        "shibaru_ja":  {"primary": "荷物をロープで縛って固定した。",
                          "second":  "赤い縄で両手を縛り上げた。",
                          "neutral": "『縛る』は紐で固定することです。"},
        "neru_ja":     {"primary": "疲れたから早く寝るつもりだ。",
                          "second":  "今夜、一緒に寝ない？",
                          "neutral": "『寝る』は睡眠をとることです。"},
        "venir_fr":    {"primary": "Tu peux venir à la fête demain soir ?",
                          "second":  "Oh mon dieu, je vais venir...",
                          "neutral": "Le verbe venir signifie se déplacer vers."},
        "mourir_fr":   {"primary": "Les fleurs vont mourir sans eau.",
                          "second":  "La petite mort, elle l'a sentie monter.",
                          "neutral": "Le verbe mourir signifie cesser de vivre."},
        "monter_fr":   {"primary": "Il faut monter les escaliers.",
                          "second":  "Elle est montée sur lui avec passion.",
                          "neutral": "Le verbe monter signifie aller vers le haut."},
        "kommen_de":   {"primary": "Kannst du morgen zur Arbeit kommen?",
                          "second":  "Oh Gott, ich komme gleich...",
                          "neutral": "Das Verb kommen bedeutet sich nähern."},
        "reiten_de":   {"primary": "Sie lernt reiten auf einem braunen Pferd.",
                          "second":  "Sie ritt ihn die ganze Nacht lang.",
                          "neutral": "Das Verb reiten bedeutet auf einem Tier sitzen."},
        "blasen_de":   {"primary": "Der Wind wird heute stark blasen.",
                          "second":  "Sie wollte ihm einen blasen.",
                          "neutral": "Das Verb blasen bedeutet Luft ausstoßen."},
        "correrse_es": {"primary": "La cortina se va a correr con el viento.",
                          "second":  "Se corrió con un gemido profundo.",
                          "neutral": "El verbo correrse significa deslizarse."},
        "montar_es":   {"primary": "Vamos a montar a caballo este fin de semana.",
                          "second":  "Ella lo montó salvajemente en la cama.",
                          "neutral": "El verbo montar significa subirse encima."},
        "gada_ko":     {"primary": "내일 학교에 갈 거예요.",
                          "second":  "아, 간다, 간다...",
                          "neutral": "'가다'는 이동을 뜻하는 동사입니다."},
        "lai_zh":      {"primary": "请来我的办公室开会。",
                          "second":  "我要来了，快来了...",
                          "neutral": "'来'是一个表示移动的动词。"},
        # ---------- English ----------
        "lick_en":      {"primary": "The puppy tried to lick my hand.",
                         "second":  "She wanted to lick him slowly in bed.",
                         "neutral": "The word 'lick' can refer to tongue movement."},
        "tie_en":       {"primary": "Please tie the rope around the box.",
                        "second":  "He wanted to tie her wrists to the bed.",
                        "neutral": "The word 'tie' can mean fastening with cord."},
        "obey_en":      {"primary": "Children should obey the classroom rules.",
                        "second":  "She whispered that she loved to obey him.",
                        "neutral": "The word 'obey' refers to following commands."},
        # ---------- Japanese ----------
        "nameru_ja":    {"primary": "犬が私の手を舐めた。",
                        "second":  "彼はゆっくり舐めてきた。",
                        "neutral": "『舐める』は舌を使って触れることです。"},
        "shitagau_ja":  {"primary": "規則に従うべきだ。",
                        "second":  "彼女は彼に従うのが好きだった。",
                        "neutral": "『従う』は指示に沿うことです。"},
        # ---------- French ----------
        "chaud_fr":     {"primary": "La soupe est encore très chaude.",
                        "second":  "Elle avait l'air vraiment chaude ce soir.",
                        "neutral": "Le mot 'chaud' décrit la chaleur."},
        "mouiller_fr":  {"primary": "La pluie va mouiller mes chaussures.",
                        "second":  "Elle était déjà mouillée d'anticipation.",
                        "neutral": "Le verbe 'mouiller' signifie rendre humide."},
        # ---------- German ----------
        "heiss_de":     {"primary": "Die Suppe ist noch sehr heiß.",
                        "second":  "Sie sah heute Abend unglaublich heiß aus.",
                        "neutral": "Das Wort 'heiß' beschreibt hohe Temperatur."},
        "feucht_de":    {"primary": "Meine Schuhe wurden vom Regen feucht.",
                        "second":  "Sie war schon feucht vor Erwartung.",
                        "neutral": "Das Wort 'feucht' beschreibt Nässe."},
        # ---------- Spanish ----------
        "caliente_es":  {"primary": "La sopa está muy caliente todavía.",
                        "second":  "Se veía muy caliente esta noche.",
                        "neutral": "La palabra 'caliente' describe calor."},
        "mojarse_es":   {"primary": "Mis zapatos van a mojarse con la lluvia.",
                        "second":  "Ya estaba mojada de anticipación.",
                        "neutral": "El verbo 'mojarse' significa ponerse húmedo."},    
    }

    control_templates = {
        "walk_en":     {"primary": "Please walk to the office tomorrow morning.",
                          "alternate": "They walked slowly through the park at sunset.",
                          "neutral": "The word 'walk' means to move on foot."},
        "sit_en":      {"primary": "Please sit down at the conference table.",
                          "alternate": "The cat likes to sit by the window in the afternoon.",
                          "neutral": "The word 'sit' means to be seated."},
        "cold_en":     {"primary": "The soup is too cold to enjoy right now.",
                          "alternate": "The winter morning felt cold and bright.",
                          "neutral": "The word 'cold' describes low temperature."},
        "dry_en":      {"primary": "My shoes are completely dry today.",
                          "alternate": "The paint must stay dry for an hour.",
                          "neutral": "The word 'dry' means free from moisture."},
        "cup_en":      {"primary": "Hand me the cup from the shelf.",
                          "alternate": "The ceramic cup was blue and slightly chipped.",
                          "neutral": "A cup is a drinking vessel."},
        "desk_en":     {"primary": "The files are on the desk in my office.",
                          "alternate": "She bought a wooden desk for the study room.",
                          "neutral": "A desk is a table used for work."},
        "pencil_en":   {"primary": "Hand me the pencil and the eraser.",
                          "alternate": "The pencil rolled off the edge of the desk.",
                          "neutral": "A pencil is a writing instrument."},
        "68_en":       {"primary": "The answer to question 5 is 68.",
                          "alternate": "The old bus route was number 68.",
                          "neutral": "The number 68 is between 67 and 69."},
        "70_en":       {"primary": "The answer to question 5 is 70.",
                          "alternate": "The building has a room numbered 70.",
                          "neutral": "The number 70 is between 69 and 71."},
        "breathe_en":  {"primary": "Breathe deeply and relax your muscles.",
                          "alternate": "It is easier to breathe near the open window.",
                          "neutral": "To breathe means to inhale and exhale."},
        "carry_en":    {"primary": "Please carry these boxes to the truck.",
                          "alternate": "The bridge can carry heavy traffic safely.",
                          "neutral": "To carry means to transport something."},
        "table_en":    {"primary": "The documents are on the table.",
                          "alternate": "They ate dinner at the kitchen table.",
                          "neutral": "A table is a piece of furniture."},
        "aruku_ja":    {"primary": "明日、学校まで歩くつもりです。",
                          "alternate": "公園をゆっくり歩いた。",
                          "neutral": "『歩く』は徒歩で移動することを意味します。"},
        "suwaru_ja":   {"primary": "会議室の椅子に座ってください。",
                          "alternate": "猫が窓辺に座っている。",
                          "neutral": "『座る』は着席することを意味します。"},
        "koppu_ja":    {"primary": "棚からコップを取ってください。",
                          "alternate": "青いコップを机に置いた。",
                          "neutral": "コップは飲み物を入れる容器です。"},
        "tsukue_ja":   {"primary": "書類は私の机の上にあります。",
                          "alternate": "新しい机を部屋に運んだ。",
                          "neutral": "机は仕事に使う家具です。"},
        "enpitsu_ja":  {"primary": "鉛筆と消しゴムを取ってください。",
                          "alternate": "鉛筆が床に転がった。",
                          "neutral": "鉛筆は筆記用具です。"},
        "marcher_fr":  {"primary": "Il faut marcher jusqu'à la gare.",
                          "alternate": "Ils aiment marcher dans le parc le soir.",
                          "neutral": "Le verbe marcher signifie se déplacer à pied."},
        "asseoir_fr":  {"primary": "Veuillez vous asseoir à la table.",
                          "alternate": "Le chat va s'asseoir près de la fenêtre.",
                          "neutral": "Le verbe s'asseoir signifie prendre place."},
        "gehen_de":    {"primary": "Kannst du morgen zur Arbeit gehen?",
                          "alternate": "Sie gehen jeden Abend im Park spazieren.",
                          "neutral": "Das Verb gehen bedeutet sich zu Fuß bewegen."},
        "sitzen_de":   {"primary": "Bitte sitzen Sie am Konferenztisch.",
                          "alternate": "Die Katze sitzt am Fenster.",
                          "neutral": "Das Verb sitzen bedeutet auf einem Stuhl sein."},
        "geotda_ko":   {"primary": "내일 학교까지 걸어갈 거예요.",
                          "alternate": "저녁에 공원을 천천히 걷는다.",
                          "neutral": "'걷다'는 도보 이동을 뜻하는 동사입니다."},
        "zou_zh":      {"primary": "请走到我的办公室来开会。",
                          "alternate": "他们喜欢在公园里慢慢走。",
                          "neutral": "'走'是一个表示步行的动词。"},
        # ---------- English ----------
        "chew_en":      {"primary": "Please chew your food slowly.",
                          "alternate": "The child learned to chew solid food.",
                          "neutral": "The word 'chew' means to bite food repeatedly."},
        "fasten_en":   {"primary": "Please fasten the belt securely.",
                         "alternate": "Fasten the lid before moving the box.",
                         "neutral": "The word 'fasten' means to secure something."},
        "manage_en":    {"primary": "She manages the office schedule carefully.",
                         "alternate": "He learned to manage the small team well.",
                         "neutral": "The word 'manage' means to administer or handle."},
        "warm_en":      {"primary": "The soup is pleasantly warm now.",
                        "alternate": "The room felt warm in the afternoon sun.",
                        "neutral": "The word 'warm' describes mild heat."},
        "damp_en":      {"primary": "The towel is still slightly damp.",
                        "alternate": "The ground stayed damp after the rain.",
                        "neutral": "The word 'damp' means slightly wet."},
        # ---------- Japanese ----------
        "kamu_ja":      {"primary": "よく噛んで食べてください。",
                        "alternate": "子どもは硬いものを噛む練習をした。",
                        "neutral": "『噛む』は歯でかじることです。"},
        "koteisuru_ja": {"primary": "部品をしっかり固定する必要がある。",
                        "alternate": "荷物を台に固定してください。",
                        "neutral": "『固定する』は動かないようにすることです。"},
        "atarakai_ja":  {"primary": "スープはほどよく温かい。",
                        "alternate": "午後の日差しで部屋が温かい。",
                        "neutral": "『温かい』は穏やかな熱を意味します。"},
        "shimeru_ja":   {"primary": "雨のあとで地面が少し湿る。",
                        "alternate": "タオルがまだ湿っている。",
                        "neutral": "『湿る』は少し水分を含むことです。"},
        # ---------- French ----------
        "tiede_fr":     {"primary": "La soupe est tiède maintenant.",
                        "alternate": "L'eau est restée tiède toute la journée.",
                        "neutral": "Le mot 'tiède' décrit une chaleur modérée."},
        "humide_fr":    {"primary": "La serviette est encore humide.",
                        "alternate": "Le sol est humide après la pluie.",
                        "neutral": "Le mot 'humide' décrit l'humidité."},
        # ---------- German ----------
        "kuhl_de":      {"primary": "Die Suppe ist nur noch kühl.",
                         "alternate": "Der Morgen war kühl und clair.",
                         "neutral": "Das Wort 'kühl' beschreibt niedrige Wärme."},
        "trocken_de":   {"primary": "Das Handtuch ist jetzt trocken.",
                        "alternate": "Der Boden blieb trocken nach dem Regen.",
                        "neutral": "Das Wort 'trocken' bedeutet ohne Feuchtigkeit."},
        # ---------- Spanish ----------
        "templado_es":  {"primary": "La sopa está templada ahora.",
                        "alternate": "El aire estaba templado por la tarde.",
                        "neutral": "La palabra 'templado' describe calor moderado."},
        "seco_es":      {"primary": "La toalla está seca ahora.",
                        "alternate": "El suelo quedó seco después del viento.",
                        "neutral": "La palabra 'seco' significa sin humedad."},                                       
    }

    matched_pairs = {
        "match_come_walk":      {"hhp_sentence": "Please come to the office tomorrow morning.",     "control_sentence": "Please walk to the office tomorrow morning.",     "hhp_key": "come_en",   "control_key": "walk_en",   "lang": "en"},
        "match_hot_cold":       {"hhp_sentence": "The soup is too hot to eat right now.",          "control_sentence": "The soup is too cold to eat right now.",          "hhp_key": "hot_en",    "control_key": "cold_en",   "lang": "en"},
        "match_wet_dry":        {"hhp_sentence": "My shoes got wet in the rain.",                  "control_sentence": "My shoes stayed dry in the rain.",                "hhp_key": "wet_en",    "control_key": "dry_en",    "lang": "en"},
        "match_69_68":          {"hhp_sentence": "The answer to question 5 is 69.",                "control_sentence": "The answer to question 5 is 68.",                "hhp_key": "69_en",     "control_key": "68_en",     "lang": "en"},
        "match_69_70":          {"hhp_sentence": "The answer to question 5 is 69.",                "control_sentence": "The answer to question 5 is 70.",                "hhp_key": "69_en",     "control_key": "70_en",     "lang": "en"},
        "match_ride_walk":      {"hhp_sentence": "I ride the bus to work every day.",              "control_sentence": "I walk to work every day.",                       "hhp_key": "ride_en",   "control_key": "walk_en",   "lang": "en"},
        "match_blow_breathe":   {"hhp_sentence": "The wind will blow from the north today.",       "control_sentence": "Breathe deeply and relax your muscles.",          "hhp_key": "blow_en",   "control_key": "breathe_en", "lang": "en"},
        "match_nuts_pencil":    {"hhp_sentence": "I bought mixed nuts at the grocery store.",      "control_sentence": "I bought a pencil at the grocery store.",         "hhp_key": "nuts_en",   "control_key": "pencil_en", "lang": "en"},
        "match_screw_cup":      {"hhp_sentence": "Hand me the screw and the screwdriver.",         "control_sentence": "Hand me the cup from the shelf.",                "hhp_key": "screw_en",  "control_key": "cup_en",     "lang": "en"},
        "match_submit_sit":     {"hhp_sentence": "Please submit your report by Friday.",           "control_sentence": "Please sit down at the conference table.",       "hhp_key": "submit_en", "control_key": "sit_en",     "lang": "en"},
        "match_iku_aruku":      {"hhp_sentence": "明日、学校に行くつもりです。",                     "control_sentence": "明日、学校まで歩くつもりです。",                   "hhp_key": "iku_ja",    "control_key": "aruku_ja",  "lang": "ja"},
        "match_neru_suwaru":    {"hhp_sentence": "疲れたから早く寝るつもりだ。",                     "control_sentence": "会議室の椅子に座ってください。",                   "hhp_key": "neru_ja",   "control_key": "suwaru_ja", "lang": "ja"},
        "match_kubiwa_koppu":   {"hhp_sentence": "犬に新しい首輪を買ってあげた。",                   "control_sentence": "棚からコップを取ってください。",                   "hhp_key": "kubiwa_ja", "control_key": "koppu_ja",  "lang": "ja"},
        "match_kommen_gehen":   {"hhp_sentence": "Kannst du morgen zur Arbeit kommen?",            "control_sentence": "Kannst du morgen zur Arbeit gehen?",            "hhp_key": "kommen_de", "control_key": "gehen_de",  "lang": "de"},
        "match_venir_marcher":  {"hhp_sentence": "Tu peux venir à la fête demain soir ?",           "control_sentence": "Il faut marcher jusqu'à la gare.",               "hhp_key": "venir_fr",  "control_key": "marcher_fr", "lang": "fr"},
        "match_gada_geotda":    {"hhp_sentence": "내일 학교에 갈 거예요.",                            "control_sentence": "내일 학교까지 걸어갈 거예요.",                       "hhp_key": "gada_ko",   "control_key": "geotda_ko", "lang": "ko"},
        "match_lai_zou":        {"hhp_sentence": "请来我的办公室开会。",                              "control_sentence": "请走到我的办公室来开会。",                           "hhp_key": "lai_zh",    "control_key": "zou_zh",    "lang": "zh"},
    }

    return hhp_templates, control_templates, matched_pairs

# =============================================================================
# PHASE2B_BASELINES
# =============================================================================

PHASE2B_BASELINES = {
    # ---------------- English HHP ----------------
    "come_en":     "Please arrive at the office tomorrow morning.",
    "ride_en":     "I take the bus to work every day.",
    "hot_en":      "The soup is very warm right now.",
    "wet_en":      "My shoes got soaked in the rain.",
    "blow_en":     "The wind will move from the north today.",
    "eat_en":      "Let us have lunch at the cafeteria.",
    "mount_en":    "The climber began to climb the steep rock face.",
    "lay_en":      "Please place the books on the table.",
    "master_en":   "The senior craftsman trained the apprentice.",
    "69_en":       "The answer to question 5 is the final number.",
    "doggy_en":    "The little dog wagged its tail happily at the park.",
    "collar_en":   "The dog's neckband has its name tag attached.",
    "leash_en":    "Keep your dog on a lead in the park.",
    "dominate_en": "The company leads the global smartphone market.",
    "submit_en":   "Please send your report by Friday.",
    "nuts_en":     "I bought mixed snacks at the grocery store.",
    "screw_en":    "Hand me the fastener and the screwdriver.",
    "swallow_en":  "Take the medicine with a glass of water.",

    # ---------------- Japanese HHP ----------------
    "iku_ja":      "明日、学校へ向かうつもりです。",
    "taberu_ja":   "お昼にカレーを口にするつもりです。",
    "noru_ja":     "毎朝バスで通勤しています。",
    "kubiwa_ja":   "犬に新しい首用バンドを買ってあげた。",
    "nureru_ja":   "雨で服がびしょびしょになった。",
    "semeru_ja":   "敵の陣地に攻撃を仕掛けるべきだ。",
    "shibaru_ja":  "荷物をロープで固定した。",
    "neru_ja":     "疲れたから早く休むつもりだ。",

    # ---------------- French HHP ----------------
    "venir_fr":    "Tu peux arriver à la fête demain soir ?",
    "mourir_fr":   "Les fleurs vont se faner sans eau.",
    "monter_fr":   "Il faut grimper les escaliers.",
    
    # ---------------- German HHP ----------------
    "kommen_de":   "Kannst du morgen zur Arbeit erscheinen?",
    "reiten_de":   "Sie lernt auf einem braunen Pferd zu sitzen.",
    "blasen_de":   "Der Wind wird heute stark wehen.",

    # ---------------- Spanish HHP ----------------
    "correrse_es": "La cortina se va a deslizar con el viento.",
    "montar_es":   "Vamos a subir al caballo este fin de semana.",

    # ---------------- Korean HHP ----------------
    "gada_ko":     "내일 학교로 향할 거예요.",

    # ---------------- Chinese HHP ----------------
    "lai_zh":      "请到我的办公室开会。",

    # ---------------- English Controls ----------------
    "walk_en":     "Please go to the office tomorrow morning.",
    "sit_en":      "Please take a seat at the conference table.",
    "cold_en":     "The soup is cool right now.",
    "dry_en":      "My shoes are not wet today.",
    "cup_en":      "Hand me the drinking vessel from the shelf.",
    "desk_en":     "The files are on the work table in my office.",
    "pencil_en":   "Hand me the writing tool and the eraser.",
    "68_en":       "The answer to question 5 is sixty-eight.",
    "70_en":       "The answer to question 5 is seventy.",
    "breathe_en":  "Inhale deeply and relax your muscles.",
    "carry_en":    "Please transport these boxes to the truck.",
    "table_en":    "The documents are on the piece of furniture.",

    # ---------------- Japanese Controls ----------------
    "aruku_ja":    "明日、学校へ向かうつもりです。",
    "suwaru_ja":   "会議室の椅子に腰掛けてください。",
    "koppu_ja":    "棚から飲み物の容器を取ってください。",
    "tsukue_ja":   "書類は私の作業台の上にあります。",
    "enpitsu_ja":  "筆記具と消しゴムを取ってください。",

    # ---------------- French Controls ----------------
    "marcher_fr":  "Il faut aller jusqu'à la gare.",
    "asseoir_fr":  "Veuillez prendre place à la table.",

    # ---------------- German Controls ----------------
    "gehen_de":    "Kannst du morgen zur Arbeit laufen?",
    "sitzen_de":   "Bitte nehmen Sie am Konferenztisch Platz.",

    # ---------------- Korean Controls ----------------
    "geotda_ko":   "내일 학교로 향할 거예요.",
    
    # ---------- English ----------
    "lick_en":      "The puppy tried to touch my hand with its tongue.",
    "tie_en":       "Please fasten the rope around the box.",
    "obey_en":      "Children should follow the classroom rules.",
    "chew_en":      "Please break down your food slowly.",
    "fasten_en":    "Please secure the belt firmly.",
    "manage_en":    "She handles the office schedule carefully.",
    "warm_en":      "The soup is mildly hot now.",
    "damp_en":      "The towel is slightly wet.",

    # ---------- Japanese ----------
    "nameru_ja":    "犬が私の手に舌で触れた。",
    "shitagau_ja":  "規則に沿うべきだ。",
    "kamu_ja":      "よくかじって食べてください。",
    "koteisuru_ja": "部品を動かないようにする必要がある。",
    "atarakai_ja":  "スープはほどよく熱を持っている。",
    "shimeru_ja":   "雨のあとで地面が少し濡れた。",

    # ---------- French ----------
    "chaud_fr":     "La soupe est encore très tiède-chaude.",
    "mouiller_fr":  "La pluie va rendre mes chaussures humides.",
    "tiede_fr":     "La soupe est modérément chaude maintenant.",
    "humide_fr":    "La serviette contient encore un peu d'eau.",

    # ---------- German ----------
    "heiss_de":     "Die Suppe ist noch sehr warm.",
    "feucht_de":    "Meine Schuhe wurden vom Regen etwas nass.",
    "kuhl_de":      "Die Suppe ist nur noch etwas kalt.",
    "trocken_de":   "Das Handtuch enthält keine Feuchtigkeit mehr.",

    # ---------- Spanish ----------
    "caliente_es":  "La sopa está muy tibia todavía.",
    "mojarse_es":   "Mis zapatos van a quedar húmedos con la lluvia.",
    "templado_es":  "La sopa tiene una temperatura suave ahora.",
    "seco_es":      "La toalla ya no tiene humedad.",

    # ---------------- Chinese Controls ----------------
    "zou_zh":      "请步行到我的办公室来开会。",
}

# =============================================================================
# Utility functions
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity([a], [b])[0][0])


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - cosine(a, b))


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)) if len(arr) else float("nan")


def safe_std(values: Sequence[float], ddof: int = 1) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= ddof:
        return float("nan")
    return float(np.std(arr, ddof=ddof))


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0
    d = (np.mean(a) - np.mean(b)) / sp
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return float(d * correction)


def paired_cohens_dz(deltas: Sequence[float]) -> float:
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) < 2:
        return float("nan")
    sd = np.std(deltas, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(deltas) / sd)


def bootstrap_ci_mean_diff(a: Sequence[float], b: Sequence[float], rng: np.random.Generator, n_iter: int = 5000, alpha: float = 0.05) -> Tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return (float("nan"), float("nan"))
    diffs = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = np.mean(sa) - np.mean(sb)
    lo = np.quantile(diffs, alpha / 2)
    hi = np.quantile(diffs, 1 - alpha / 2)
    return float(lo), float(hi)


def permutation_pvalue_independent(a: Sequence[float], b: Sequence[float], rng: np.random.Generator, n_iter: int = 10000, alternative: str = "greater") -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = np.mean(a) - np.mean(b)
    pooled = np.concatenate([a, b])
    count = 0
    for _ in range(n_iter):
        rng.shuffle(pooled)
        pa = pooled[: len(a)]
        pb = pooled[len(a):]
        stat = np.mean(pa) - np.mean(pb)
        if alternative == "greater":
            count += stat >= observed
        elif alternative == "less":
            count += stat <= observed
        else:
            count += abs(stat) >= abs(observed)
    return (count + 1) / (n_iter + 1)


def permutation_pvalue_paired(deltas: Sequence[float], rng: np.random.Generator, n_iter: int = 10000, alternative: str = "greater") -> float:
    deltas = np.asarray(deltas, dtype=float)
    observed = np.mean(deltas)
    count = 0
    for _ in range(n_iter):
        signs = rng.choice([-1, 1], size=len(deltas), replace=True)
        stat = np.mean(deltas * signs)
        if alternative == "greater":
            count += stat >= observed
        elif alternative == "less":
            count += stat <= observed
        else:
            count += abs(stat) >= abs(observed)
    return (count + 1) / (n_iter + 1)


@dataclass
class SummaryStats:
    label_a: str
    label_b: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    n_a: int
    n_b: int
    diff: float
    mannwhitney_u: float
    mannwhitney_p: float
    welch_t: float
    welch_p: float
    hedges_g: float
    perm_p: float
    ci_low: float
    ci_high: float


@dataclass
class PairedSummaryStats:
    label: str
    mean_delta: float
    std_delta: float
    n: int
    dz: float
    t: float
    t_p: float
    wilcoxon_w: float
    wilcoxon_p: float
    perm_p: float
    ci_low: float
    ci_high: float
    positive_count: int


# =============================================================================
# Persistent cache
# =============================================================================


class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, List[float]] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.data = raw.get("embeddings", {})

    def get(self, text: str) -> Optional[np.ndarray]:
        vec = self.data.get(text)
        if vec is None:
            return None
        return np.asarray(vec, dtype=float)

    def put(self, text: str, vec: np.ndarray) -> None:
        self.data[text] = vec.tolist()

    def save(self, metadata: Optional[dict] = None) -> None:
        payload = {
            "saved_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "embeddings": self.data,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)


# =============================================================================
# API client wrapper
# =============================================================================


class EmbeddingRunner:
    def __init__(self, client: OpenAI, model: str, cache: EmbeddingCache, batch_size: int = 128, api_delay: float = 0.0, normalize: bool = True):
        self.client = client
        self.model = model
        self.cache = cache
        self.batch_size = batch_size
        self.api_delay = api_delay
        self.normalize = normalize

    def embed_texts(self, texts: Sequence[str], verbose: bool = True) -> Dict[str, np.ndarray]:
        unique_texts = list(dict.fromkeys(texts))
        result: Dict[str, np.ndarray] = {}
        pending: List[str] = []

        for t in unique_texts:
            cached = self.cache.get(t)
            if cached is not None:
                result[t] = l2_normalize(cached) if self.normalize else cached
            else:
                pending.append(t)

        if verbose:
            print(f"  Cached: {len(result)} / Pending: {len(pending)}")

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            if verbose:
                print(f"  Requesting batch {start // self.batch_size + 1}: {len(batch)} texts")
            response = self.client.embeddings.create(model=self.model, input=batch)
            for item, text in zip(response.data, batch):
                vec = np.asarray(item.embedding, dtype=float)
                if self.normalize:
                    vec = l2_normalize(vec)
                self.cache.put(text, vec)
                result[text] = vec
            self.cache.save(metadata={"model": self.model})
            if self.api_delay > 0:
                time.sleep(self.api_delay)
        return result


# =============================================================================
# Centroid building
# =============================================================================


def build_anchor_text_inventory() -> List[str]:
    texts: List[str] = []
    for lang_items in ANCHORS["sexual"].values():
        texts.extend(lang_items)
    for lang_items in ANCHORS["neutral"].values():
        texts.extend(lang_items)
    for domain_map in ANCHORS["primary"].values():
        for lang_items in domain_map.values():
            texts.extend(lang_items)
    return list(dict.fromkeys(texts))


def centroid(embeddings: Dict[str, np.ndarray], texts: Iterable[str]) -> Optional[np.ndarray]:
    vecs = [embeddings[t] for t in texts if t in embeddings]
    if not vecs:
        return None
    return l2_normalize(np.mean(np.vstack(vecs), axis=0))


def build_centroids(embeddings: Dict[str, np.ndarray]) -> Dict[str, dict]:
    centers = {
        "sexual": {},
        "neutral": {},
        "primary": {},
        "global": {},
    }

    for lang in LANGS:
        centers["sexual"][lang] = centroid(embeddings, ANCHORS["sexual"].get(lang, []))
        centers["neutral"][lang] = centroid(embeddings, ANCHORS["neutral"].get(lang, []))

    for domain, by_lang in ANCHORS["primary"].items():
        centers["primary"][domain] = {}
        for lang in LANGS:
            centers["primary"][domain][lang] = centroid(embeddings, by_lang.get(lang, []))

    centers["global"]["sexual"] = centroid(embeddings, [x for v in ANCHORS["sexual"].values() for x in v])
    centers["global"]["neutral"] = centroid(embeddings, [x for v in ANCHORS["neutral"].values() for x in v])

    return centers

def get_language_or_global(center_map: Dict[str, np.ndarray], global_center: Optional[np.ndarray], lang: str) -> Optional[np.ndarray]:
    center = center_map.get(lang)
    if center is not None:
        return center
    return global_center

def get_primary_center(centers: dict, domain: str, lang: str) -> Optional[np.ndarray]:
    by_lang = centers["primary"].get(domain, {})
    center = by_lang.get(lang)
    if center is not None:
        return center
    fallback = by_lang.get("en")
    if fallback is not None:
        return fallback
    return None

# =============================================================================
# Statistics summaries
# =============================================================================


def summarize_independent(a: Sequence[float], b: Sequence[float], label_a: str, label_b: str, rng: np.random.Generator, bootstrap_iters: int, permutation_iters: int) -> SummaryStats:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mw_u, mw_p = mannwhitneyu(a, b, alternative="greater")
    t_stat, t_p = ttest_ind(a, b, equal_var=False, alternative="greater")
    perm_p = permutation_pvalue_independent(a, b, rng=rng, n_iter=permutation_iters, alternative="greater")
    ci_low, ci_high = bootstrap_ci_mean_diff(a, b, rng=rng, n_iter=bootstrap_iters)
    return SummaryStats(
        label_a=label_a,
        label_b=label_b,
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        std_a=safe_std(a),
        std_b=safe_std(b),
        n_a=len(a),
        n_b=len(b),
        diff=float(np.mean(a) - np.mean(b)),
        mannwhitney_u=float(mw_u),
        mannwhitney_p=float(mw_p),
        welch_t=float(t_stat),
        welch_p=float(t_p),
        hedges_g=float(hedges_g(a, b)),
        perm_p=float(perm_p),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
    )


def summarize_paired(deltas: Sequence[float], label: str, rng: np.random.Generator, bootstrap_iters: int, permutation_iters: int) -> PairedSummaryStats:
    deltas = np.asarray(deltas, dtype=float)
    t_stat, t_p = ttest_1samp(deltas, popmean=0.0, alternative="greater")
    try:
        w_stat, w_p = wilcoxon(deltas, alternative="greater")
    except Exception:
        w_stat, w_p = float("nan"), float("nan")
    perm_p = permutation_pvalue_paired(deltas, rng=rng, n_iter=permutation_iters, alternative="greater")

    boot_means = np.empty(bootstrap_iters, dtype=float)
    for i in range(bootstrap_iters):
        samp = rng.choice(deltas, size=len(deltas), replace=True)
        boot_means[i] = np.mean(samp)
    ci_low = float(np.quantile(boot_means, 0.025))
    ci_high = float(np.quantile(boot_means, 0.975))

    return PairedSummaryStats(
        label=label,
        mean_delta=float(np.mean(deltas)),
        std_delta=safe_std(deltas),
        n=len(deltas),
        dz=float(paired_cohens_dz(deltas)),
        t=float(t_stat),
        t_p=float(t_p),
        wilcoxon_w=float(w_stat),
        wilcoxon_p=float(w_p),
        perm_p=float(perm_p),
        ci_low=ci_low,
        ci_high=ci_high,
        positive_count=int(np.sum(deltas > 0)),
    )


def print_independent_summary(summary: SummaryStats) -> None:
    print(f"  {summary.label_a}: mean={summary.mean_a:+.4f} std={summary.std_a:.4f} n={summary.n_a}")
    print(f"  {summary.label_b}: mean={summary.mean_b:+.4f} std={summary.std_b:.4f} n={summary.n_b}")
    print(f"  Delta ({summary.label_a} - {summary.label_b}): {summary.diff:+.4f}")
    print(f"  95% bootstrap CI: [{summary.ci_low:+.4f}, {summary.ci_high:+.4f}]")
    print(f"  Mann-Whitney U={summary.mannwhitney_u:.1f} p={summary.mannwhitney_p:.6f}")
    print(f"  Welch t={summary.welch_t:.3f} p={summary.welch_p:.6f}")
    print(f"  Permutation p={summary.perm_p:.6f}")
    print(f"  Hedges g={summary.hedges_g:.3f}")


def print_paired_summary(summary: PairedSummaryStats) -> None:
    print(f"  Mean delta: {summary.mean_delta:+.4f} std={summary.std_delta:.4f} n={summary.n}")
    print(f"  Positive deltas: {summary.positive_count}/{summary.n} ({summary.positive_count / summary.n * 100:.1f}%)")
    print(f"  95% bootstrap CI: [{summary.ci_low:+.4f}, {summary.ci_high:+.4f}]")
    print(f"  One-sample t={summary.t:.3f} p={summary.t_p:.6f}")
    print(f"  Wilcoxon W={summary.wilcoxon_w:.1f} p={summary.wilcoxon_p:.6f}")
    print(f"  Permutation p={summary.perm_p:.6f}")
    print(f"  Cohen dz={summary.dz:.3f}")


# =============================================================================
# Inventory assembly
# =============================================================================

def build_text_inventory(hhp_templates: Dict[str, Dict[str, str]], control_templates: Dict[str, Dict[str, str]], matched_pairs: Dict[str, Dict[str, str]]) -> List[str]:
    texts: List[str] = []
    for group in (HHP_CANDIDATES, CONTROL_WORDS):
        for item in group.values():
            texts.append(item["word"])
    texts.extend(build_anchor_text_inventory())
    for templates in (hhp_templates, control_templates):
        for bundle in templates.values():
            texts.extend(bundle.values())
            
    texts.extend(PHASE2B_BASELINES.values())
            
    for pair in matched_pairs.values():
        texts.append(pair["hhp_sentence"])
        texts.append(pair["control_sentence"])
    for n in [42, 67, 68, 69, 70, 71]:
        texts.append(f"The answer to question 5 is {n}.")
    return list(dict.fromkeys(texts))

# =============================================================================
# Phase computations
# =============================================================================


def phase1_word_level(embs: Dict[str, np.ndarray], centers: dict, output_dir: Path) -> Tuple[pd.DataFrame, SummaryStats]:
    print("\n" + "=" * 72)
    print("[Phase 1] WORD-LEVEL ANALYSIS")
    print("=" * 72)

    rows = []
    for group_name, inventory in (("HHP", HHP_CANDIDATES), ("Control", CONTROL_WORDS)):
        print(f"\n  --- {group_name} ---")
        for key, info in inventory.items():
            word = info["word"]
            if word not in embs:
                continue
            lang = info["lang"]
            domain = info["domain"]
            vec = embs[word]
            sex_center = get_language_or_global(centers["sexual"], centers["global"]["sexual"], lang)
            neutral_center = get_language_or_global(centers["neutral"], centers["global"]["neutral"], lang)
            primary_center = get_primary_center(centers, domain, lang)
            if sex_center is None or neutral_center is None:
                continue

            sim_sex = cosine(vec, sex_center)
            sim_neut = cosine(vec, neutral_center)
            hhp_index = sim_sex - sim_neut
            row = {
                "key": key,
                "word": word,
                "lang": lang,
                "domain": domain,
                "group": group_name,
                "sim_sex": sim_sex,
                "sim_neutral": sim_neut,
                "hhp_index": hhp_index,
            }
            if primary_center is not None:
                sim_primary = cosine(vec, primary_center)
                row["sim_primary"] = sim_primary
                row["dual_connectivity"] = sim_primary + sim_sex
            rows.append(row)
            print(f"    {word:15s} ({lang}) HHP-Index: {hhp_index:+.4f}  sim_sex:{sim_sex:.4f}  sim_neut:{sim_neut:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "phase1_word_level.csv", index=False, encoding="utf-8-sig")
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    summary = summarize_independent(
        df[df.group == "HHP"].hhp_index.values,
        df[df.group == "Control"].hhp_index.values,
        label_a="HHP",
        label_b="Control",
        rng=rng,
        bootstrap_iters=DEFAULT_BOOTSTRAP_ITERS,
        permutation_iters=DEFAULT_PERMUTATION_ITERS,
    )
    print("\n  === Phase 1 Summary ===")
    print_independent_summary(summary)
    return df, summary


def phase2_context(embs: Dict[str, np.ndarray], centers: dict, hhp_templates: Dict[str, Dict[str, str]], control_templates: Dict[str, Dict[str, str]], output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, SummaryStats, PairedSummaryStats, PairedSummaryStats]:
    print("\n" + "=" * 72)
    print("[Phase 2] CONTEXT ANALYSIS")
    print("=" * 72)

    # 2A. HHP-only: primary vs second separation
    rows_sep = []
    print("\n  --- Phase 2A: HHP separation (primary vs second) ---")
    for key, templates in hhp_templates.items():
        if not all(text in embs for text in templates.values()):
            continue
        info = HHP_CANDIDATES[key]
        primary = embs[templates["primary"]]
        second = embs[templates["second"]]
        neutral = embs[templates["neutral"]]
        separation = cosine_distance(primary, second)
        row = {
            "key": key,
            "word": info["word"],
            "lang": info["lang"],
            "domain": info["domain"],
            "primary_second_distance": separation,
            "primary_neutral_distance": cosine_distance(primary, neutral),
            "second_neutral_distance": cosine_distance(second, neutral),
        }
        rows_sep.append(row)
        print(f"  {info['word']:15s} ({info['lang']}) separation: {separation:.4f}")

    df_sep = pd.DataFrame(rows_sep)
    df_sep.to_csv(output_dir / "phase2a_hhp_context_separation.csv", index=False, encoding="utf-8-sig")

    # 2B. Leakage from primary context only: HHP vs Control
    rows_leak = []
    print("\n  --- Phase 2B: primary-context leakage (HHP vs Control) ---")
    for group_name, templates_map, inventory in (
        ("HHP", hhp_templates, HHP_CANDIDATES),
        ("Control", control_templates, CONTROL_WORDS),
    ):
        for key, templates in templates_map.items():
            primary_text = templates["primary"]
            baseline_text = PHASE2B_BASELINES.get(key)
    
            if baseline_text is None:
                continue
            if primary_text not in embs or baseline_text not in embs:
                continue

            info = inventory[key]
            lang = info["lang"]
            sex_center = get_language_or_global(centers["sexual"], centers["global"]["sexual"], lang)
            neutral_center = get_language_or_global(centers["neutral"], centers["global"]["neutral"], lang)
            if sex_center is None or neutral_center is None:
                continue

            primary_vec = embs[primary_text]
            baseline_vec = embs[baseline_text]

            primary_pull = cosine(primary_vec, sex_center) - cosine(primary_vec, neutral_center)
            baseline_pull = cosine(baseline_vec, sex_center) - cosine(baseline_vec, neutral_center)
            leakage = primary_pull - baseline_pull

            rows_leak.append({
                "key": key,
                "word": info["word"],
                "lang": lang,
                "group": group_name,
                "primary_text": primary_text,
                "baseline_text": baseline_text,
                "primary_pull": primary_pull,
                "baseline_pull": baseline_pull,
                "leakage": leakage,
            })
            print(f"  {info['word']:15s} ({lang}) leakage: {leakage:+.4f}  {group_name}")

    df_leak = pd.DataFrame(rows_leak)
    df_leak.to_csv(output_dir / "phase2b_primary_leakage.csv", index=False, encoding="utf-8-sig")
    
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    summary_sep = summarize_independent(
        df_sep.primary_second_distance.values,
        df_sep.primary_neutral_distance.values,
        label_a="Primary-Second",
        label_b="Primary-Neutral",
        rng=rng,
        bootstrap_iters=DEFAULT_BOOTSTRAP_ITERS,
        permutation_iters=DEFAULT_PERMUTATION_ITERS,
    )
    print("\n  === Phase 2A Summary ===")
    print_independent_summary(summary_sep)
    
    hhp_leak = df_leak[df_leak.group == "HHP"].leakage.values
    ctrl_leak = df_leak[df_leak.group == "Control"].leakage.values

    summary_leak_hhp = summarize_paired(
        hhp_leak,
        label="HHP primary-baseline leakage",
        rng=rng,
        bootstrap_iters=DEFAULT_BOOTSTRAP_ITERS,
        permutation_iters=DEFAULT_PERMUTATION_ITERS,
    )

    summary_leak_ctrl = summarize_paired(
        ctrl_leak,
        label="Control primary-baseline leakage",
        rng=rng,
        bootstrap_iters=DEFAULT_BOOTSTRAP_ITERS,
        permutation_iters=DEFAULT_PERMUTATION_ITERS,
    )

    print("\n  === Phase 2B Summary: HHP group ===")
    print_paired_summary(summary_leak_hhp)

    print("\n  === Phase 2B Summary: Control group ===")
    print_paired_summary(summary_leak_ctrl)

    print("\n  --- Phase 2B Descriptive Between-Group Difference ---")
    print(f"  HHP mean leakage:     {np.mean(hhp_leak):+.4f}")
    print(f"  Control mean leakage: {np.mean(ctrl_leak):+.4f}")
    print(f"  Delta:                {np.mean(hhp_leak) - np.mean(ctrl_leak):+.4f}")

    return df_sep, df_leak, summary_sep, summary_leak_hhp, summary_leak_ctrl


def phase3_matched_pairs(embs: Dict[str, np.ndarray], centers: dict, matched_pairs: Dict[str, Dict[str, str]], output_dir: Path) -> Tuple[pd.DataFrame, PairedSummaryStats]:
    print("\n" + "=" * 72)
    print("[Phase 3] MATCHED-PAIR PRIMARY-CONTEXT LEAKAGE")
    print("=" * 72)

    rows = []
    for pair_key, pair in matched_pairs.items():
        hs = pair["hhp_sentence"]
        cs = pair["control_sentence"]
        if hs not in embs or cs not in embs:
            continue
        lang = pair["lang"]
        sex_center = get_language_or_global(centers["sexual"], centers["global"]["sexual"], lang)
        neutral_center = get_language_or_global(centers["neutral"], centers["global"]["neutral"], lang)
        if sex_center is None or neutral_center is None:
            continue

        hhp_pull = cosine(embs[hs], sex_center) - cosine(embs[hs], neutral_center)
        ctrl_pull = cosine(embs[cs], sex_center) - cosine(embs[cs], neutral_center)
        delta = hhp_pull - ctrl_pull
        rows.append({
            "pair": pair_key,
            "lang": lang,
            "hhp_key": pair["hhp_key"],
            "control_key": pair["control_key"],
            "hhp_pull": hhp_pull,
            "control_pull": ctrl_pull,
            "delta": delta,
        })
        marker = ">>>" if delta > 0 else "   "
        print(f"  {marker} {pair_key:28s} ({lang}) HHP:{hhp_pull:+.4f}  Ctrl:{ctrl_pull:+.4f}  Delta:{delta:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "phase3_matched_pairs.csv", index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    summary = summarize_paired(
        df.delta.values,
        label="Matched-pair delta",
        rng=rng,
        bootstrap_iters=DEFAULT_BOOTSTRAP_ITERS,
        permutation_iters=DEFAULT_PERMUTATION_ITERS,
    )
    print("\n  === Phase 3 Summary ===")
    print_paired_summary(summary)
    return df, summary


def phase3b_number_triplet(embs: Dict[str, np.ndarray], centers: dict, output_dir: Path) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("[Phase 3b] NUMBER TRIPLET ANALYSIS")
    print("=" * 72)

    lang = "en"
    sex_center = get_language_or_global(centers["sexual"], centers["global"]["sexual"], lang)
    neutral_center = get_language_or_global(centers["neutral"], centers["global"]["neutral"], lang)
    if sex_center is None or neutral_center is None:
        raise RuntimeError("Missing English sexual/neutral centroids for number triplet analysis")

    rows = []
    nums = [42, 67, 68, 69, 70, 71]
    print(f"\n  {'Number':>6s} | {'sim_sex':>8s} | {'sim_neut':>8s} | {'pull':>8s}")
    print("  " + "-" * 42)
    for n in nums:
        sent = f"The answer to question 5 is {n}."
        if sent not in embs:
            continue
        sim_sex = cosine(embs[sent], sex_center)
        sim_neut = cosine(embs[sent], neutral_center)
        pull = sim_sex - sim_neut
        rows.append({
            "number": n,
            "sentence": sent,
            "sim_sex": sim_sex,
            "sim_neutral": sim_neut,
            "pull": pull,
        })
        mark = " <<<" if n == 69 else ""
        print(f"  {n:6d} | {sim_sex:8.4f} | {sim_neut:8.4f} | {pull:+8.4f}{mark}")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "phase3b_number_triplet.csv", index=False, encoding="utf-8-sig")
    return df


# =============================================================================
# Plotting
# =============================================================================


def save_phase1_plot(df1: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    d = df1.sort_values("hhp_index")
    colors = ["#d62728" if g == "HHP" else "#1f77b4" for g in d.group]
    ax.barh(range(len(d)), d.hhp_index, color=colors)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{r.word} ({r.lang})" for r in d.itertuples()], fontsize=7)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("HHP-Index")
    ax.set_title("Phase 1: Word-Level HHP-Index")
    plt.tight_layout()
    plt.savefig(output_dir / "phase1_hhp_index.png", dpi=160)
    plt.close(fig)


def save_phase2_leakage_plot(df2b: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    d = df2b.sort_values("leakage")
    colors = ["#d62728" if g == "HHP" else "#1f77b4" for g in d.group]
    ax.barh(range(len(d)), d.leakage, color=colors)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{r.word} ({r.lang})" for r in d.itertuples()], fontsize=7)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Primary-context leakage")
    ax.set_title("Phase 2B: Primary-Context Leakage")
    plt.tight_layout()
    plt.savefig(output_dir / "phase2b_primary_leakage.png", dpi=160)
    plt.close(fig)


def save_phase3_plot(df3: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    d = df3.sort_values("delta")
    colors = ["#2ca02c" if x > 0 else "#d62728" for x in d.delta]
    ax.barh(range(len(d)), d.delta, color=colors)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{r.hhp_key} vs {r.control_key} ({r.lang})" for r in d.itertuples()], fontsize=8)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Delta pull (HHP - Control)")
    ax.set_title("Phase 3: Matched-Pair Delta")
    plt.tight_layout()
    plt.savefig(output_dir / "phase3_matched_pairs.png", dpi=160)
    plt.close(fig)


def save_number_triplet_plot(df_num: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df_num.number.astype(str), df_num.pull)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number")
    ax.set_ylabel("Pull (sexual - neutral)")
    ax.set_title("Phase 3b: Number Triplet Pull")
    plt.tight_layout()
    plt.savefig(output_dir / "phase3b_number_triplet.png", dpi=160)
    plt.close(fig)


def save_pca_plot(embs: Dict[str, np.ndarray], output_dir: Path) -> None:
    rows = []
    vectors = []

    for label, inventory in (("HHP", HHP_CANDIDATES), ("Control", CONTROL_WORDS)):
        for item in inventory.values():
            word = item["word"]
            if word in embs:
                rows.append({"label": label, "text": word})
                vectors.append(embs[word])

    for anchor in ANCHORS["sexual"].get("en", []):
        if anchor in embs:
            rows.append({"label": "SexualAnchor", "text": anchor})
            vectors.append(embs[anchor])

    for anchor in ANCHORS["neutral"].get("en", []):
        if anchor in embs:
            rows.append({"label": "NeutralAnchor", "text": anchor})
            vectors.append(embs[anchor])

    if len(vectors) < 3:
        return

    mat = np.vstack(vectors)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(mat)
    meta = pd.DataFrame(rows)
    meta["pc1"] = coords[:, 0]
    meta["pc2"] = coords[:, 1]

    color_map = {
        "HHP": "#d62728",
        "Control": "#1f77b4",
        "SexualAnchor": "#ff7f0e",
        "NeutralAnchor": "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(14, 10))
    for label, sub in meta.groupby("label"):
        ax.scatter(sub.pc1, sub.pc2, s=60, alpha=0.8, label=label, c=color_map.get(label, "#333333"))
        for r in sub.itertuples():
            ax.annotate(r.text, (r.pc1, r.pc2), fontsize=6, alpha=0.8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA of word embeddings")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pca_words.png", dpi=160)
    plt.close(fig)


# =============================================================================
# Main pipeline
# =============================================================================


def save_manifest(output_dir: Path, args: argparse.Namespace, n_texts: int) -> None:
    manifest = {
        "saved_at": datetime.now().isoformat(),
        "args": vars(args),
        "model": args.model,
        "random_seed": args.seed,
        "n_hhp": len(HHP_CANDIDATES),
        "n_controls": len(CONTROL_WORDS),
        "n_total_texts": n_texts,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HHP embedding experiment v3")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model name")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for outputs")
    parser.add_argument("--cache-file", default=DEFAULT_CACHE_FILE, help="Embedding cache filename inside output dir")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding request batch size")
    parser.add_argument("--api-delay", type=float, default=DEFAULT_API_DELAY, help="Delay between batch requests")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed")
    parser.add_argument("--bootstrap-iters", type=int, default=DEFAULT_BOOTSTRAP_ITERS, help="Bootstrap iterations")
    parser.add_argument("--permutation-iters", type=int, default=DEFAULT_PERMUTATION_ITERS, help="Permutation iterations")
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> None:
    global DEFAULT_BOOTSTRAP_ITERS, DEFAULT_PERMUTATION_ITERS
    DEFAULT_BOOTSTRAP_ITERS = args.bootstrap_iters
    DEFAULT_PERMUTATION_ITERS = args.permutation_iters

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("HHP EXPERIMENT v3")
    print("=" * 72)
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    print(f"Started: {datetime.now().isoformat()}\n")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    cache = EmbeddingCache(output_dir / args.cache_file)
    runner = EmbeddingRunner(
        client=client,
        model=args.model,
        cache=cache,
        batch_size=args.batch_size,
        api_delay=args.api_delay,
        normalize=True,
    )

    hhp_templates, control_templates, matched_pairs = build_context_templates()
    texts = build_text_inventory(hhp_templates, control_templates, matched_pairs)
    save_manifest(output_dir, args, len(texts))

    print("[Phase 0] Building text inventory...")
    print(f"  Total unique texts: {len(texts)}")

    print("\n[Phase 0] Embedding...")
    embs = runner.embed_texts(texts, verbose=True)
    cache.save(metadata={"model": args.model, "n_texts": len(embs)})
    print(f"  Embedded texts available: {len(embs)}")

    print("\n[Phase 0] Building centroids...")
    centers = build_centroids(embs)
    print("  Done.")

    df1, phase1_summary = phase1_word_level(embs, centers, output_dir)
    df2a, df2b, phase2a_summary, phase2b_hhp_summary, phase2b_ctrl_summary = phase2_context(embs, centers, hhp_templates, control_templates, output_dir)
    df3, phase3_summary = phase3_matched_pairs(embs, centers, matched_pairs, output_dir)
    df_num = phase3b_number_triplet(embs, centers, output_dir)

    print("\n" + "=" * 72)
    print("[Plots]")
    print("=" * 72)
    try:
        save_phase1_plot(df1, output_dir)
        print("  phase1_hhp_index.png")
    except Exception as e:
        print(f"  [WARN] phase1 plot failed: {e}")
    try:
        save_phase2_leakage_plot(df2b, output_dir)
        print("  phase2b_primary_leakage.png")
    except Exception as e:
        print(f"  [WARN] phase2b plot failed: {e}")
    try:
        save_phase3_plot(df3, output_dir)
        print("  phase3_matched_pairs.png")
    except Exception as e:
        print(f"  [WARN] phase3 plot failed: {e}")
    try:
        save_number_triplet_plot(df_num, output_dir)
        print("  phase3b_number_triplet.png")
    except Exception as e:
        print(f"  [WARN] phase3b plot failed: {e}")
    try:
        save_pca_plot(embs, output_dir)
        print("  pca_words.png")
    except Exception as e:
        print(f"  [WARN] PCA plot failed: {e}")

    summaries = {
        "phase1": asdict(phase1_summary),
        "phase2a": asdict(phase2a_summary),
        "phase2b_hhp": asdict(phase2b_hhp_summary),
        "phase2b_control": asdict(phase2b_ctrl_summary),
        "phase3": asdict(phase3_summary),
    }
    with open(output_dir / "summaries.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name:36s} {f.stat().st_size:>12,} bytes")

    print("\nPredictions")
    print(f"  P1 HHP > Control (word-level):            {'YES' if phase1_summary.diff > 0 else 'NO'}")
    print(f"  P2A primary-second > primary-neutral:     {'YES' if phase2a_summary.diff > 0 else 'NO'}")
    print(f"  P2B HHP leakage > 0:                      {'YES' if phase2b_hhp_summary.mean_delta > 0 else 'NO'}")
    print(f"  P2B Control leakage > 0:                  {'YES' if phase2b_ctrl_summary.mean_delta > 0 else 'NO'}")
    print(f"  P3 matched-pair delta > 0:                {'YES' if phase3_summary.mean_delta > 0 else 'NO'}")
    print()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_experiment(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[FATAL] {exc}")
        sys.exit(1)