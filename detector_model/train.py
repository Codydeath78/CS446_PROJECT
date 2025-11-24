import os

os.environ["CROWS_LOCAL_PATH"] = "/content/crows_pairs_anonymized.csv"

#set dataset size limits
os.environ["MAX_PER_DATASET"] = "50000"
os.environ["MAX_BIOS"] = "20000"

#nudge class balance/mix
os.environ["TARGET_POS_FRAC"] = "0.5"
os.environ["MIX_CROWS"]  = "0.25"
os.environ["MIX_STEREO"] = "0.20"
os.environ["MIX_BIOS"]   = "0.20"
os.environ["MIX_CIVIL"] = "0.07"
os.environ["MIX_HATEX"] = "0.03"
#rest filled by civil and hatex

#pairwise training stays active with these hyperparams
os.environ["PAIR_LAMBDA"]        = "0.08"
os.environ["PAIR_LAMBDA_FINAL"]  = "0.05"
os.environ["PAIR_MARGIN"]        = "0.5"
os.environ["PAIR_BS"]            = "16"

#calibration safety
os.environ["CAL_T_MIN"] = "0.7"
os.environ["CAL_T_MAX"] = "3.0"
os.environ["USE_CLASS_WEIGHTS"] = "1"


#the purpose of batchencoding is to patch the to() method to ignore non_blocking argument
from transformers.tokenization_utils_base import BatchEncoding
_orig_to = BatchEncoding.to
def _to_ignore_kwargs(self, *args, **kwargs):
    kwargs.pop("non_blocking", None)
    return _orig_to(self, *args, **kwargs)
BatchEncoding.to = _to_ignore_kwargs
print("Patched BatchEncoding.to() to ignore non_blocking.")


import re
import platform
IS_WINDOWS = platform.system() == "Windows"
import json
import random
import numpy as np
from typing import List, Dict, Optional, Tuple

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import snapshot_download

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from scipy.special import softmax

#GPU optimizations
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

#configurable hyperparameters
MODEL_NAME   = "microsoft/deberta-v3-base"
OUT_DIR      = "./out"
SAVE_DIR     = "./model"
MAX_LEN      = 256
EPOCHS       = 4
TRAIN_BS     = 16
EVAL_BS      = 32
LR           = 2e-5
SEED         = 42
MAX_PER_DATASET = int(os.getenv("MAX_PER_DATASET", "50000"))

#deployment threshold (after calibration)
DEPLOY_THRESHOLD = float(os.getenv("DEPLOY_THRESHOLD", "0.65"))

#class balance after TRAIN rebalancing (undersample majority)
TARGET_POS_FRAC = float(os.getenv("TARGET_POS_FRAC", "0.5"))
BALANCE_STEREOSET = os.getenv("BALANCE_STEREOSET", "1") == "1"
BALANCE_CROWS = os.getenv("BALANCE_CROWS", "1") == "1"
BALANCE_BIOS = os.getenv("BALANCE_BIOS", "0") == "1"

#hard cap for Bios in TRAIN
MAX_BIOS = os.getenv("MAX_BIOS")
MAX_BIOS = int(MAX_BIOS) if MAX_BIOS is not None and MAX_BIOS.strip() != "" else None

#source mix proportions (post-rebalancing, pre-tokenization)
MIX_CROWS  = float(os.getenv("MIX_CROWS", "0.35"))
MIX_STEREO = float(os.getenv("MIX_STEREO", "0.20"))
#strengthen Bios presence in TRAIN mix (data-level presence)
MIX_BIOS   = float(os.getenv("MIX_BIOS", "0.35"))
MIX_CIVIL = float(os.getenv("MIX_CIVIL", "0.07"))
MIX_HATEX = float(os.getenv("MIX_HATEX", "0.03"))

#pairwise loss hyperparams
PAIR_LAMBDA = float(os.getenv("PAIR_LAMBDA", "0.08"))
PAIR_LAMBDA_FINAL = float(os.getenv("PAIR_LAMBDA_FINAL", "0.05"))
PAIR_MARGIN = float(os.getenv("PAIR_MARGIN", "0.5"))
PAIR_BS = int(os.getenv("PAIR_BS", "16"))
PAIR_USE_PROB_INIT = float(os.getenv("PAIR_USE_PROB_INIT", "0.4"))
PAIR_USE_PROB_FINAL = float(os.getenv("PAIR_USE_PROB_FINAL", "0.9"))
PAIR_HARD_MINING = os.getenv("PAIR_HARD_MINING", "1") == "1"
PAIR_HARD_MULT = int(os.getenv("PAIR_HARD_MULT", "2"))
PAIR_HARD_WITH_NO_GRAD = os.getenv("PAIR_HARD_WITH_NO_GRAD", "1") == "1"

#eval / calibration
EVAL_FORCE_BIOS_POS       = int(os.getenv("EVAL_FORCE_BIOS_POS", "0"))
EVAL_FORCE_BIOS_POS_FRAC  = float(os.getenv("EVAL_FORCE_BIOS_POS_FRAC", "0.006"))
CALIBRATE_PAIRWISE_METRIC = os.getenv("CALIBRATE_PAIRWISE_METRIC", "1") == "1"

#bios pre/post split rebalancing
BIB_TARGET_POS_FRAC = float(os.getenv("BIB_TARGET_POS_FRAC", "0.0"))
BIB_RESAMPLE_MODE   = os.getenv("BIB_RESAMPLE_MODE", "undersample_neg")
BIB_REBALANCE_STAGE = os.getenv("BIB_REBALANCE_STAGE", "train_only")

#target fractions
BIB_TRAIN_TARGET_POS_FRAC = float(os.getenv("BIB_TRAIN_TARGET_POS_FRAC", "0.50"))
EVAL_BIOS_TARGET_POS_FRAC = float(os.getenv("EVAL_BIOS_TARGET_POS_FRAC", "0.50"))

#bios lightweight positive mining
BIB_MIN_POS_FRAC = float(os.getenv("BIB_MIN_POS_FRAC", "0.006"))
BIB_MAX_PROMOTE  = int(os.getenv("BIB_MAX_PROMOTE", "200"))
BIB_VERBOSE      = os.getenv("BIB_VERBOSE", "1") == "1"

#calibration safety
CAL_T_MIN = float(os.getenv("CAL_T_MIN", "0.7"))
CAL_T_MAX = float(os.getenv("CAL_T_MAX", "3.0"))
CAL_FAILSAFE_MIN_POS_RATE = float(os.getenv("CAL_FAILSAFE_MIN_POS_RATE", "0.02"))

BALANCE_CIVIL = os.getenv("BALANCE_CIVIL", "1") == "1"
BALANCE_HATEX = os.getenv("BALANCE_HATEX", "1") == "1"

#extra datasets for positive signal
from datasets import load_dataset as hf_load

#data collator for CrowS-Pairs
class CrowsCollator:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __call__(self, batch):
        return crows_collate(batch, self.tokenizer)

#CrowS-Pairs collate function
def load_civil_comments_subset(max_rows=100000):
    #civil comments toxicity detection
    try:
        ds = hf_load("civil_comments", split="train")
        df = ds.to_pandas()
        def gt(df, col, thr=0.5):
            return (df[col] > thr) if col in df.columns else pd.Series(False, index=df.index)
        pos = (
            gt(df, "toxicity") |
            gt(df, "severe_toxicity") |
            gt(df, "identity_attack") |
            gt(df, "insult") |
            gt(df, "obscene") |
            gt(df, "threat")
        )
        keep = df.sample(n=min(len(df), max_rows), random_state=SEED)
        rows = []
        for _, r in keep.iterrows():
            txt = str(r.get("text", "")).strip()
            if not txt:
                continue
            rows.append({"text": txt, "label": int(bool(pos.loc[r.name])), "source": "civil"})
        return Dataset.from_list(rows)
    except Exception as e:
        print("[civil_comments] skipped:", e)
        return None

#hatexplain subset loader
def load_hatexplain_subset(max_rows=60000):
    #hatexplain hate speech detection
    try:
        ds = hf_load("hatexplain", split="train")
        df = ds.to_pandas()
        #majority vote function
        def maj(lbls):
            if not isinstance(lbls, list): return 0
            from collections import Counter
            c = Counter(lbls)
            lab, _ = c.most_common(1)[0]
            return 1 if lab in ("offensive","hatespeech","hate_speech","hateful") else 0
        rows = []
        #sample max_rows
        for _, r in df.sample(n=min(len(df), max_rows), random_state=SEED).iterrows():
            tokens = r.get("post_tokens", [])
            txt = " ".join(tokens) if isinstance(tokens, list) else str(r.get("post", ""))
            if not txt: continue
            y = maj(r.get("annotators", {}).get("label", [])) if isinstance(r.get("annotators"), dict) else maj(r.get("label", []))
            rows.append({"text": txt, "label": int(y), "source": "hatex"})
        return Dataset.from_list(rows)
    except Exception as e:
        print("[hatexplain] skipped:", e)
        return None

#set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#ID mapping
SRC2ID = {"crows": 0, "stereoset": 1, "bios": 2, "civil": 3, "hatex": 4}
def _src_to_id_list(src_list):
    return [SRC2ID.get(s, -1) for s in src_list]

#get first available key from list
def _get_any(ex: Dict, keys: List[str], default=""):
    for k in keys:
        if k in ex and ex[k] is not None:
            return ex[k]
    return default

#temperature scaling (calibration)
class _TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(()))  # T = exp(log_T)
    def forward(self, logits):
        return logits / self.log_T.exp().clamp_min(1e-4)

#fit temperature scaling on validation logits and labels
def fit_temperature(logits_np, labels_np, max_iter=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.tensor(logits_np, dtype=torch.float32, device=device)
    labels = torch.tensor(labels_np, dtype=torch.long, device=device)
    scaler = _TempScaler().to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    nll = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        loss = nll(scaler(logits), labels)
        loss.backward()
        return loss
    opt.step(closure)
    T = float(scaler.log_T.exp().detach().cpu())
    return T

#bias-in-Bios rules
STEREO_PHRASES = [
    "women are ", "men are ", "girls are ", "boys are ",
    "because she is a woman", "because he is a man",
    "female drivers", "male nurses", "as a woman,", "as a man,",
    "women can't", "men can't", "women should", "men should",
    "women must", "men must",
    "women are too emotional", "men are logical", "women are better caregivers",
    "men are breadwinners", "female scientists", "male kindergarten teachers",
    "women should stay home", "men should not cry",
    "girls aren't good at math", "boys will be boys", "women can't code", "men can't be nurses",
]
#regex for bios mining
_BIB_GENDER_TOKENS = r"(?:\bwomen?\b|\bmen\b|\bgirls?\b|\bboys?\b|\bfemale\b|\bmale\b|\bshe\b|\bhe\b)"
_BIB_JUDGMENT      = r"(?:\bshould(?:n't)?\b|\bmust\b|\bcan(?:not|'t)?\b|\baren't\b|\bisn't\b|\bnever\b|\balways\b|\bbetter\b|\bworse\b|\bnot (?:good|fit|suited)\b|\bgood at\b|\bbad at\b|\bsupposed to\b)"
_BIB_STEREO_COMBO  = re.compile(fr"{_BIB_GENDER_TOKENS}.*{_BIB_JUDGMENT}|{_BIB_JUDGMENT}.*{_BIB_GENDER_TOKENS}", re.IGNORECASE)
_BIB_EXTRA_PHRASES = [
    "as a woman", "as a man",
    "female engineers", "male nurses",
    "women in the kitchen", "men are breadwinners",
    "women aren't good at", "men aren't good at",
    "women should", "men should",
    "women can't", "men can't",
]

#bios detection
def _bios_is_positive(txt: str) -> bool:
    if not isinstance(txt, str): return False
    t = " " + txt.lower() + " "
    if any(p in t for p in STEREO_PHRASES): return True
    if any(p in t for p in _BIB_EXTRA_PHRASES): return True
    if _BIB_STEREO_COMBO.search(txt): return True
    return False

#rebalance a source within a Dataset to target positive fraction
def rebalance_source_to_pos_frac(ds: Dataset, source_name: str, target_pos_frac: float, seed: int = SEED,
                                 mode: str = "undersample_neg") -> Dataset:
    if ds is None or len(ds) == 0 or target_pos_frac <= 0.0 or target_pos_frac >= 1.0 or mode != "undersample_neg":
        return ds
    df = ds.to_pandas()
    mask = (df["source"] == source_name)
    block = df[mask]
    if block.empty:
        return ds
    lab = block["label"].to_numpy()
    pos_idx = block.index[lab == 1]
    neg_idx = block.index[lab == 0]
    n_pos   = len(pos_idx)
    n_neg   = len(neg_idx)
    if n_pos == 0:
        print(f"[rebalance_source_to_pos_frac] No positives in '{source_name}', cannot rebalance.")
        return ds
    n_neg_keep = int(round(n_pos * (1.0 - target_pos_frac) / max(1e-8, target_pos_frac)))
    n_neg_keep = min(n_neg_keep, n_neg)
    rng = np.random.default_rng(seed)
    keep_neg_idx = rng.choice(neg_idx, size=n_neg_keep, replace=False) if n_neg_keep > 0 else np.array([], dtype=int)
    keep_idx = np.sort(np.concatenate([pos_idx, keep_neg_idx]))
    kept_block = df.loc[keep_idx]
    other = df[~mask]
    new_df = pd.concat([other, kept_block], ignore_index=True).sample(frac=1.0, random_state=seed)
    new_ds = Dataset.from_pandas(new_df[["text", "label", "source"]], preserve_index=False)
    nb = new_df[new_df["source"] == source_name]
    frac = float((nb["label"] == 1).mean()) if len(nb) else 0.0
    print(f"[rebalance_source_to_pos_frac] '{source_name}' -> pos_frac ≈ {frac:.3f} "
          f"(kept pos={len(pos_idx)} neg={len(keep_neg_idx)})")
    return new_ds

#CrowS-Pairs loader and converter
def load_crows_pairs_df() -> pd.DataFrame:
    local_path = os.getenv("CROWS_LOCAL_PATH", "").strip()
    if local_path:
        if not os.path.exists(local_path):
            raise RuntimeError(f"CROWS_LOCAL_PATH set but file not found: {local_path}")
        print(f"[CrowS-Pairs] using local file: {local_path}")
        if local_path.lower().endswith((".json",".jsonl")):
            try:
                return pd.read_json(local_path, lines=local_path.lower().endswith(".jsonl"))
            except ValueError:
                rows = []
                with open(local_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
                return pd.DataFrame(rows)
        sep = "\t" if local_path.lower().endswith(".tsv") else ","
        return pd.read_csv(local_path, sep=sep, encoding="utf-8", engine="python")
    try:
        ds = load_dataset("nyu-mll/crows_pairs")
        split = "train" if "train" in ds else list(ds.keys())[0]
        print("[CrowS-Pairs] available splits:", list(ds.keys()))
        print("[CrowS-Pairs] columns:", ds[split].column_names)
        return ds[split].to_pandas()
    except Exception as e:
        print("[CrowS-Pairs] load_dataset failed (expected):", e)
    print("[CrowS-Pairs] no data files; skipping CrowS for now.")
    return pd.DataFrame()

#convert CrowS-Pairs dataframe to Datasets
def convert_crows_pairs_to_dataset(df: pd.DataFrame) -> Optional[Dataset]:
    if df is None or df.empty:
        return None
    stereo_col = None
    anti_col   = None
    for c in df.columns:
        lc = c.lower().replace("_", "-")
        if lc in ["sent-more","sent_more","stereotype","stereotypical","stereotype-text","stereotype_text"]:
            stereo_col = c if stereo_col is None else stereo_col
        if lc in ["sent-less","sent_less","anti-stereotype","anti_stereotype","anti"]:
            anti_col = c if anti_col is None else anti_col
    if stereo_col is None or anti_col is None:
        print("[CrowS-Pairs] expected columns not found; skipping.")
        return None
    rows = []
    for _, r in df.iterrows():
        if isinstance(r.get(stereo_col), str):
            rows.append({"text": r[stereo_col], "label": 1, "source": "crows"})
        if isinstance(r.get(anti_col), str):
            rows.append({"text": r[anti_col], "label": 0, "source": "crows"})
    if not rows:
        print("[CrowS-Pairs] mapping produced 0 rows; skipping.")
    if len(rows) > MAX_PER_DATASET:
        rows = rows[:MAX_PER_DATASET]
    return Dataset.from_list(rows) if rows else None

#extract CrowS-Pairs stereo and anti-stereo texts
def extract_crows_pair_texts(df: pd.DataFrame) -> Optional[Tuple[List[str], List[str]]]:
    if df is None or df.empty:
        return None
    stereo_col = None
    anti_col   = None
    for c in df.columns:
        lc = c.lower().replace("_", "-")
        if lc in ["sent-more","sent_more","stereotype","stereotypical","stereotype-text","stereotype_text"]:
            stereo_col = stereo_col or c
        if lc in ["sent-less","sent_less","anti-stereotype","anti_stereotype","anti"]:
            anti_col = anti_col or c
    if stereo_col is None or anti_col is None:
        return None
    return df[stereo_col].astype(str).tolist(), df[anti_col].astype(str).tolist()

#stereoSet loader and converter
def load_stereoset_frames() -> pd.DataFrame:
    repo_id = "McGill-NLP/stereoset"
    all_rows = []
    for config in ["intrasentence", "intersentence"]:
        try:
            ds = load_dataset(repo_id, config)
            for split_name, split_ds in ds.items():
                print(f"[StereoSet/{config}] split: {split_name} columns:", split_ds.column_names)
                pdf = split_ds.to_pandas()
                pdf["__config__"] = config
                pdf["__split__"] = split_name
                all_rows.append(pdf)
        except Exception as e:
            print(f"[StereoSet/{config}] load_dataset failed; will try snapshot: {e}")
    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    print("[StereoSet] Falling back to snapshot & manual parse…")
    local_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
    json_files = []
    for root, _, files in os.walk(local_dir):
        for f in files:
            if f.lower().endswith(".json"):
                json_files.append(os.path.join(root, f))
    if not json_files:
        raise RuntimeError("StereoSet: no JSON files in snapshot.")
    dfs = []
    for path in json_files:
        with open(path, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except Exception:
                fh.seek(0)
                data = [json.loads(line) for line in fh if line.strip()]
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            norm = pd.json_normalize(data["data"])
        elif isinstance(data, list):
            norm = pd.json_normalize(data)
        else:
            continue
        norm["__src__"] = os.path.basename(path)
        dfs.append(norm)
    if not dfs:
        raise RuntimeError("StereoSet: unusable JSON content.")
    df = pd.concat(dfs, ignore_index=True)
    print("[StereoSet] detected columns:", list(df.columns))
    return df

#convert StereoSet dataframe to Dataset
def convert_stereoset_to_dataset(df: pd.DataFrame) -> Optional[Dataset]:
    if df is None or df.empty:
        return None

    # StereoSet numeric labels:
    # 0 = stereotype (POSITIVE), 1 = anti-stereotype (NEGATIVE), 2 = unrelated, 3 = other/unknown
    def norm_label_numeric(v):
        try:
            iv = int(v)
        except Exception:
            return None
        if iv == 0:
            return 1  #1 = biased/stereotype
        if iv == 1:
            return 0  #0 = anti-stereotype (non-biased)
        return None   # 2/3 so drop
    #stereoSet string labels:
    def norm_label_string(s):
        if not isinstance(s, str):
            return None
        t = s.strip().lower().replace("_", "-")
        if t in {"stereotype", "stereo", "stereotypical", "sent-more", "sent-more-bias"}:
            return 1
        if t in {"anti-stereotype", "anti", "sent-less", "sent-less-bias"}:
            return 0
        return None

    rows = []
    #iterate over rows
    for _, r in df.iterrows():
        context = _get_any(r, ["context", "sentence", "target", "txt"])
        sents = r.get("sentences")

        #list-of-dicts variant
        if isinstance(sents, list):
            for item in sents:
                if not isinstance(item, dict):
                    continue
                text = item.get("sentence") or item.get("text") or item.get("value") or ""
                lab  = (norm_label_string(item.get("label")) if "label" in item else
                        norm_label_string(item.get("gold_label")) if "gold_label" in item else None)
                if not lab and "label" in item and isinstance(item["label"], (int, np.integer)):
                    lab = norm_label_numeric(item["label"])
                if isinstance(text, str) and text.strip() and lab is not None:
                    rows.append({"text": f"{(context or '').strip()} {text}".strip(),
                                 "label": lab, "source": "stereoset"})
            continue

        #dict with parallel arrays
        if isinstance(sents, dict):
            sent_list = None
            for k in ("sentence", "sentences", "text", "value"):
                if k in sents and isinstance(sents[k], (list, np.ndarray, pd.Series)):
                    sent_list = list(sents[k]); break
            if not sent_list:
                continue

            n = len(sent_list)

            #gold_label if it matches length
            gold = sents.get("gold_label")
            if isinstance(gold, (list, np.ndarray, pd.Series)) and len(gold) == n:
                for text, gl in zip(sent_list, list(gold)):
                    if not isinstance(text, str) or not text.strip():
                        continue
                    lab = norm_label_numeric(gl) if isinstance(gl, (int, np.integer)) else norm_label_string(gl)
                    if lab is not None:
                        rows.append({"text": f"{(context or '').strip()} {text}".strip(),
                                     "label": lab, "source": "stereoset"})
                continue

            #fallback case: majority from per-annotator labels[i]['label'] (arrays of 0/1/2/3)
            ann = sents.get("labels")
            if isinstance(ann, (list, np.ndarray, pd.Series)) and len(ann) == n:
                for text, per_item in zip(sent_list, list(ann)):
                    if not isinstance(text, str) or not text.strip() or not isinstance(per_item, dict):
                        continue
                    lab_arr = per_item.get("label")
                    if isinstance(lab_arr, (list, np.ndarray, pd.Series)) and len(lab_arr) > 0:
                        #majority vote among annotators, drop 2/3 before voting if possible
                        arr = [int(x) for x in lab_arr if int(x) in (0,1)]
                        if not arr:
                            continue
                        maj = 0 if arr.count(0) >= arr.count(1) else 1
                        lab = norm_label_numeric(maj)
                        if lab is not None:
                            rows.append({"text": f"{(context or '').strip()} {text}".strip(),
                                         "label": lab, "source": "stereoset"})
                continue

            #else: unknown sub-shape so skip
            continue

        #else: unknown 'sentences' type so skip

    if not rows:
        print("[StereoSet] mapping produced 0 rows; check label alignment (gold_label) or annotator arrays.")
        return None

    random.shuffle(rows)
    if len(rows) > MAX_PER_DATASET:
        rows = rows[:MAX_PER_DATASET]
    return Dataset.from_list(rows)


#Bias in Bios loader and converter
def load_bias_in_bios_df() -> pd.DataFrame:
    repo_id = "LabHC/bias_in_bios"
    try:
        ds = load_dataset(repo_id, split="train")
        print("[BiasInBios] columns:", ds.column_names)
        return ds.to_pandas()
    except Exception as e:
        print("[BiasInBios] load_dataset failed; snapshot fallback:", e)
    local_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
    candidates = []
    for root, _, files in os.walk(local_dir):
        for f in files:
            low = f.lower()
            if low.endswith((".parquet", ".csv", ".json", ".jsonl")):
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise RuntimeError("Bias-in-Bios: no data files found in snapshot.")
    candidates.sort(key=lambda p: (p.lower().endswith(".parquet"), p.lower().endswith(".csv")), reverse=True)
    path = candidates[0]
    print(f"[BiasInBios] parsing file: {path}")
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path, encoding="utf-8", engine="python")
    else:
        try:
            df = pd.read_json(path, lines=path.lower().endswith(".jsonl"))
        except ValueError:
            rows = []
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            df = pd.DataFrame(rows)
    print("[BiasInBios] detected columns:", list(df.columns))
    return df

#convert Bias-in-Bios dataframe to Dataset
def convert_bios_to_dataset(df: pd.DataFrame) -> Optional[Dataset]:
    if df is None or df.empty:
        return None
    text_col = None
    for cand in ["bio", "hard_text", "text", "content"]:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        texty = [c for c in df.columns if df[c].dtype == object]
        if not texty:
            print("[BiasInBios] no text column; skipping.")
            return None
        text_col = max(texty, key=lambda c: df[c].astype(str).str.len().mean())
    texts = df[text_col].astype(str).tolist()
    labels = [1 if _bios_is_positive(t) else 0 for t in texts]
    pos0 = int(sum(labels))
    n    = len(labels)
    frac0 = pos0 / max(1, n)
    if BIB_VERBOSE:
        print(f"[Bios map] initial positives={pos0}/{n} ({frac0:.4%}) via conservative rules")
    if frac0 < BIB_MIN_POS_FRAC and BIB_MIN_POS_FRAC > 0.0:
        need = int(round(BIB_MIN_POS_FRAC * n)) - pos0
        need = max(0, min(need, BIB_MAX_PROMOTE))
        if need > 0:
            cand_idx = []
            for i, t in enumerate(texts):
                if labels[i] == 1:
                    continue
                if _BIB_STEREO_COMBO.search(t):
                    cand_idx.append(i)
            if cand_idx:
                rng = np.random.default_rng(SEED)
                take = min(need, len(cand_idx))
                promote = rng.choice(cand_idx, size=take, replace=False)
                for i in promote:
                    labels[i] = 1
                if BIB_VERBOSE:
                    print(f"[Bios map] promoted {take} additional bios -> positive (target ~{BIB_MIN_POS_FRAC:.4%})")
    rows = [{"text": t, "label": int(y), "source": "bios"} for t, y in zip(texts, labels)]
    random.shuffle(rows)
    if len(rows) > MAX_PER_DATASET:
        rows = rows[:MAX_PER_DATASET]
    ds = Dataset.from_list(rows)
    if BIB_VERBOSE:
        lab = np.array(ds["label"])
        print(f"[Bios map] final positives={int((lab==1).sum())}/{len(ds)} ({(lab==1).mean():.4%})")
    return ds

#stratified train/eval split
def stratified_split(ds: Dataset, test_size: float = 0.1, seed: int = SEED) -> Tuple[Dataset, Dataset]:
    if ds is None or len(ds) == 0:
        return None, None
    labels = np.array(ds["label"])
    idx = np.arange(len(labels))
    if len(np.unique(labels)) < 2:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        cut = int(round(len(idx) * (1 - test_size)))
        train_idx, eval_idx = idx[:cut], idx[cut:]
    else:
        train_idx, eval_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=labels
        )
    return ds.select(sorted(train_idx)), ds.select(sorted(eval_idx))

#balance positives and negatives within a source
def balance_within_source(ds: Dataset, seed=SEED) -> Dataset:
    if ds is None or len(ds) == 0:
        return ds
    lab = np.array(ds["label"])
    pos_idx = np.where(lab == 1)[0]
    neg_idx = np.where(lab == 0)[0]
    m = min(len(pos_idx), len(neg_idx))
    if m == 0:
        return ds
    rng = np.random.default_rng(seed)
    pos_keep = rng.choice(pos_idx, size=m, replace=False)
    neg_keep = rng.choice(neg_idx, size=m, replace=False)
    keep = np.sort(np.concatenate([pos_keep, neg_keep]))
    print(f"[balance_within_source] kept pos={len(pos_keep)} neg={len(neg_keep)} (size={len(keep)})")
    return ds.select(keep)

#Build combined dataset from all sources
def build_combined():
    print("=== Loading raw datasets ===")
    crows_df  = load_crows_pairs_df()
    stereo_df = load_stereoset_frames()
    bios_df   = load_bias_in_bios_df()
    civil_ds  = load_civil_comments_subset(max_rows=80000)
    hatex_ds  = load_hatexplain_subset(max_rows=40000)


    #inspect the first 'sentences' payload for shape
    try:
        sample_sentences = stereo_df.iloc[0]["sentences"]
        print("[StereoSet] sample sentences type:", type(sample_sentences))
        if isinstance(sample_sentences, dict):
            print("[StereoSet] sample sentences keys:", list(sample_sentences.keys()))
            for k, v in sample_sentences.items():
                if isinstance(v, (list, np.ndarray, pd.Series)):
                    print(f"[StereoSet] key {k} -> list len={len(v)} first={v[0] if len(v)>0 else None}")
                else:
                    print(f"[StereoSet] key {k} -> type={type(v)}")
        elif isinstance(sample_sentences, list):
            print("[StereoSet] sample list length:", len(sample_sentences))
            if sample_sentences and isinstance(sample_sentences[0], dict):
                print("[StereoSet] first item keys:", list(sample_sentences[0].keys()))
    except Exception as _e:
        print("[StereoSet] inspector failed:", _e)

    #pre-split label balancing within each source to avoid negative floods
    if BALANCE_CIVIL and civil_ds is not None:
        civil_ds = balance_within_source(civil_ds, SEED)
    if BALANCE_HATEX and hatex_ds is not None:
        hatex_ds = balance_within_source(hatex_ds, SEED)


    if not crows_df.empty:
        print("[CrowS-Pairs] head():\n", crows_df.head(3))
    print("[StereoSet]    head():\n", stereo_df.head(3))
    print("[BiasInBios]   head():\n", bios_df.head(3))
    if civil_ds is not None and len(civil_ds) > 0:
        print("[CivilComments] sample:\n", civil_ds[:3])
    if hatex_ds is not None and len(hatex_ds) > 0:
        print("[HateXplain]   sample:\n", hatex_ds[:3])

    crows_ds  = convert_crows_pairs_to_dataset(crows_df)
    stereo_ds = convert_stereoset_to_dataset(stereo_df)
    bios_ds   = convert_bios_to_dataset(bios_df)

    if stereo_ds is None or len(stereo_ds) == 0:
        raise RuntimeError("StereoSet parser still empty — please print a full example row.")
    else:
        y = np.array(stereo_ds["label"])
        print("[StereoSet] parsed examples:", len(stereo_ds),
              "pos(stereo)=", int((y==1).sum()), "neg(anti)=", int((y==0).sum()))

    if BIB_REBALANCE_STAGE == "pre_split" and BIB_TRAIN_TARGET_POS_FRAC > 0.0 and bios_ds is not None:
        bios_ds = rebalance_source_to_pos_frac(bios_ds, "bios", BIB_TRAIN_TARGET_POS_FRAC, SEED, "undersample_neg")

    if BALANCE_STEREOSET and stereo_ds is not None:
        stereo_ds = balance_within_source(stereo_ds, SEED)
    if BALANCE_CROWS and crows_ds is not None:
        crows_ds = balance_within_source(crows_ds, SEED)
    if BALANCE_BIOS and bios_ds is not None:
        bios_ds = balance_within_source(bios_ds, SEED)
    #perform stratified train/eval split per dataset
    def stratified(ds):
        return stratified_split(ds) if ds is not None else (None, None)

    civil_tr, civil_ev = stratified(civil_ds)
    hatex_tr, hatex_ev = stratified(hatex_ds)
    crows_tr,  crows_ev  = stratified_split(crows_ds)
    stereo_tr, stereo_ev = stratified_split(stereo_ds)
    bios_tr,   bios_ev   = stratified_split(bios_ds)

    train_parts = [d for d in [crows_tr, stereo_tr, bios_tr] if d is not None and len(d) > 0]
    eval_parts  = [d for d in [crows_ev, stereo_ev, bios_ev] if d is not None and len(d) > 0]
    train_parts += [d for d in [civil_tr, hatex_tr] if d is not None and len(d) > 0]
    eval_parts  += [d for d in [civil_ev, hatex_ev] if d is not None and len(d) > 0]
    if not train_parts or not eval_parts:
        raise RuntimeError("No train/eval data after per-dataset split.")

    train_ds = train_parts[0]
    for d in train_parts[1:]:
        train_ds = concatenate_datasets([train_ds, d])
    eval_ds = eval_parts[0]
    for d in eval_parts[1:]:
        eval_ds = concatenate_datasets([eval_ds, d])

    train_ds = train_ds.shuffle(seed=SEED)
    eval_ds  = eval_ds.shuffle(seed=SEED)

    print("[Combined] train:", len(train_ds), "eval:", len(eval_ds))
    #print label and source distributions
    def summarize(ds, name):
        labs = np.array(ds["label"])
        pos = int((labs == 1).sum()); neg = int((labs == 0).sum())
        print(f"[{name}] label dist: pos={pos} neg={neg} (pos_rate={pos/len(labs):.3f})")
        srcs = pd.Series(ds["source"]).value_counts()
        print(f"[{name}] source dist:\n{srcs}")

    summarize(train_ds, "TRAIN (pre-balance)")
    summarize(eval_ds,  "EVAL")

    crows_pairs = extract_crows_pair_texts(crows_df)
    return train_ds, eval_ds, crows_pairs

#rebalance training set to target positive fraction
def rebalance_train(ds: Dataset, target_pos_frac: float = TARGET_POS_FRAC, seed: int = SEED) -> Dataset:
    labels = np.array(ds["label"])
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    n = len(labels)
    target_pos = int(round(n * target_pos_frac))
    target_neg = n - target_pos
    target_pos = min(target_pos, len(idx_pos))
    target_neg = min(target_neg, len(idx_neg))
    if abs(len(idx_pos) - target_pos) < 10 and abs(len(idx_neg) - target_neg) < 10:
        return ds
    rng = np.random.default_rng(seed)
    sel_pos = rng.choice(idx_pos, size=target_pos, replace=False) if target_pos > 0 else np.array([], dtype=int)
    sel_neg = rng.choice(idx_neg, size=target_neg, replace=False) if target_neg > 0 else np.array([], dtype=int)
    keep_idx = np.sort(np.concatenate([sel_pos, sel_neg]))
    ds_bal = ds.select(keep_idx)
    print(f"[TRAIN balance] from pos={len(idx_pos)}/neg={len(idx_neg)} to pos={target_pos}/neg={target_neg} "
          f"(pos_rate={target_pos/max(1,(target_pos+target_neg)):.3f})")
    return ds_bal

#rebalance any dataset to target positive fraction which allows downsampling
def rebalance_to_pos_frac(ds: Dataset, target_pos_frac: float, seed: int = SEED) -> Dataset:
    if ds is None or len(ds) == 0:
        return ds
    labels = np.array(ds["label"])
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng = np.random.default_rng(seed)

    if abs(target_pos_frac - 0.5) < 1e-6:
        m = min(len(pos_idx), len(neg_idx))
        if m == 0:
            return ds
        sel_pos = rng.choice(pos_idx, size=m, replace=False)
        sel_neg = rng.choice(neg_idx, size=m, replace=False)
        keep_idx = np.sort(np.concatenate([sel_pos, sel_neg]))
        print(f"[rebalance_to_pos_frac] exact 50/50 -> pos={m} neg={m} (size={2*m})")
        return ds.select(keep_idx)

    #otherwise keep original target fraction, but allow downsizing
    n = len(labels)
    target_pos = int(round(n * target_pos_frac))
    target_pos = min(target_pos, len(pos_idx))
    target_neg = min(n - target_pos, len(neg_idx))
    sel_pos = rng.choice(pos_idx, size=target_pos, replace=False) if target_pos > 0 else np.array([], dtype=int)
    sel_neg = rng.choice(neg_idx, size=target_neg, replace=False) if target_neg > 0 else np.array([], dtype=int)
    keep_idx = np.sort(np.concatenate([sel_pos, sel_neg]))
    print(f"[rebalance_to_pos_frac] pos={len(sel_pos)} neg={len(sel_neg)} "
          f"(pos_rate={len(sel_pos)/max(1,(len(sel_pos)+len(sel_neg))):.3f}, size={len(keep_idx)})")
    return ds.select(keep_idx)

#rebalance evaluation set per source to target positive fraction
def rebalance_eval_per_source(ds: Dataset, target_pos_frac: float = 0.5, seed: int = SEED) -> Dataset:
    if ds is None or len(ds) == 0:
        return ds
    df = ds.to_pandas()
    rng = np.random.default_rng(seed)
    blocks = []
    for src, block in df.groupby("source"):
        labels = block["label"].to_numpy()
        pos_idx = block.index[labels == 1]
        neg_idx = block.index[labels == 0]

        if target_pos_frac == 0.5:
            #exact 50/50 by downsampling the majority to the size of the minority
            m = min(len(pos_idx), len(neg_idx))
            if m == 0:
                #keep as-is to avoid empty source
                blocks.append(block)
                print(f"[eval per-source] {src}: pos={len(pos_idx)} neg={len(neg_idx)} (UNCHANGED)")
                continue
            keep_pos = rng.choice(pos_idx, size=m, replace=False)
            keep_neg = rng.choice(neg_idx, size=m, replace=False)
            keep = np.sort(np.concatenate([keep_pos, keep_neg]))
            sub = df.loc[keep]
            print(f"[eval per-source] {src}: pos={m} neg={m} size={len(sub)} (50/50)")
            blocks.append(sub)
        else:
            #aim for target fraction but respect availability
            n_pos_avail, n_neg_avail = len(pos_idx), len(neg_idx)
            #desired totals if unconstrained
            desired_total = n_pos_avail + n_neg_avail
            desired_pos = int(round(desired_total * target_pos_frac))
            desired_neg = desired_total - desired_pos
            #clamp by availability, then re-equalize to preserve ratio if one side clips
            desired_pos = min(desired_pos, n_pos_avail)
            desired_neg = min(desired_neg, n_neg_avail)
            #if still imbalanced after clamping, scale both down to keep ratio close
            if target_pos_frac == 0.5:
                k = min(desired_pos, desired_neg)
                desired_pos = desired_neg = k
            keep_pos = rng.choice(pos_idx, size=desired_pos, replace=False) if desired_pos > 0 else []
            keep_neg = rng.choice(neg_idx, size=desired_neg, replace=False) if desired_neg > 0 else []
            keep = np.sort(np.concatenate([keep_pos, keep_neg]))
            sub = df.loc[keep]
            print(f"[eval per-source] {src}: pos={len(keep_pos)} neg={len(keep_neg)} size={len(sub)} (~{target_pos_frac:.2f})")
            blocks.append(sub)

    new_df = pd.concat(blocks, ignore_index=True).sample(frac=1.0, random_state=seed)
    return Dataset.from_pandas(new_df[["text","label","source"]], preserve_index=False)

#source mix and bios cap
def cap_bios(ds: Dataset, max_bios: int, seed: int = SEED) -> Dataset:
    if ds is None or len(ds) == 0 or max_bios is None:
        return ds
    df = ds.to_pandas()
    bios_idx = df[df["source"] == "bios"].index.values
    if len(bios_idx) <= max_bios:
        return ds
    rng = np.random.default_rng(seed)
    keep_bios = rng.choice(bios_idx, size=max_bios, replace=False)
    non_bios_idx = df[df["source"] != "bios"].index.values
    keep = np.sort(np.concatenate([keep_bios, non_bios_idx]))
    df_cap = df.loc[keep].sample(frac=1.0, random_state=seed)
    return Dataset.from_pandas(df_cap[["text","label","source"]], preserve_index=False)

#resample training set to target source mix with stratified sampling within each source
def mix_sources(train_ds: Dataset, target_mix=None, seed: int = SEED) -> Dataset:
    """
    resample train to approximate a source mix, but preserve label balance within each source
    by stratified sampling
    """
    if target_mix is None:
        target_mix = {"crows": MIX_CROWS, "stereoset": MIX_STEREO, "bios": MIX_BIOS, "civil": 0.25, "hatex": 0.10}
    df = train_ds.to_pandas()
    n_total = len(df)
    rng = np.random.default_rng(seed)
    parts = []
    #iterate over sources in target mix
    for src, frac in target_mix.items():
        want = int(round(n_total * frac))
        pool = df[df["source"] == src]
        if pool.empty or want <= 0:
            continue
        pos_pool = pool[pool["label"] == 1]
        neg_pool = pool[pool["label"] == 0]
        half = want // 2
        take_pos = min(len(pos_pool), half)
        take_neg = min(len(neg_pool), want - take_pos)
        if take_pos + take_neg < want:
            remain = want - (take_pos + take_neg)
            extra_from_pos = min(remain, max(0, len(pos_pool) - take_pos))
            take_pos += extra_from_pos
            remain -= extra_from_pos
            extra_from_neg = min(remain, max(0, len(neg_pool) - take_neg))
            take_neg += extra_from_neg
        if take_pos == 0 and take_neg == 0:
            continue
        sub_parts = []
        if take_pos > 0:
            sub_parts.append(pos_pool.sample(n=take_pos, random_state=seed))
        if take_neg > 0:
            sub_parts.append(neg_pool.sample(n=take_neg, random_state=seed))
        if sub_parts:
            parts.append(pd.concat(sub_parts, ignore_index=True))
    if not parts:
        return train_ds
    new_df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    return Dataset.from_pandas(new_df[["text", "label", "source"]], preserve_index=False)

#tokenization function
def tokenize(train_ds: Dataset, eval_ds: Dataset, tokenizer):
    #map source strings to IDs
    def tok(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        enc["source_id"] = _src_to_id_list(batch["source"])
        return enc
    train_tok = train_ds.map(tok, batched=True)
    eval_tok  = eval_ds.map(tok,  batched=True)
    train_tok.set_format(type="torch", columns=["input_ids","attention_mask","label","source_id"])
    eval_tok.set_format(type="torch",  columns=["input_ids","attention_mask","label","source_id"])
    return train_tok, eval_tok

#crowS pairwise mini-dataloader and collate function
class CrowSPairDataset(TorchDataset):
    def __init__(self, stereo_texts: List[str], anti_texts: List[str]):
        assert len(stereo_texts) == len(anti_texts)
        self.s_more = stereo_texts
        self.s_less = anti_texts
    def __len__(self):
        return len(self.s_more)
    def __getitem__(self, idx):
        return self.s_more[idx], self.s_less[idx]

#crowS collate function
def crows_collate(batch, tokenizer):
    stereo, anti = zip(*batch)
    t_more = tokenizer(list(stereo), truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    t_less = tokenizer(list(anti),   truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    return t_more, t_less

#weighted Trainer with class-weighted CE and pairwise hinge loss
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, pair_loader: Optional[DataLoader]=None,
                 pair_lambda: float=0.08, pair_lambda_final: float=0.05, pair_margin: float=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.pair_loader = pair_loader
        self.pair_lambda_init = float(pair_lambda)
        self.pair_lambda_final = float(pair_lambda_final)
        self.pair_margin = float(pair_margin)

        #initialize iterator so compute_loss can actually use it
        self._pair_iter = iter(self.pair_loader) if self.pair_loader is not None else None

    #get current pairwise lambda based on progress and warmup
    def _current_pair_lambda(self):
        try:
            gs = float(self.state.global_step)
            ms = float(max(1, self.state.max_steps))
            prog = min(1.0, max(0.0, gs / ms))
            warm = float(getattr(self.args, "warmup_ratio", 0.0))
            if prog < warm:
                return 0.0
            return float(self.pair_lambda_init + (self.pair_lambda_final - self.pair_lambda_init) * prog)
        except Exception:
            return float(self.pair_lambda_init)
    #get current pairwise use probability based on progress and warmup
    def _current_pair_use_prob(self):
        try:
            gs = float(self.state.global_step)
            ms = float(max(1, self.state.max_steps))
            prog = min(1.0, max(0.0, gs / ms))
            warm = float(getattr(self.args, "warmup_ratio", 0.0))
            if prog < warm:
                return 0.0
            adj_prog = (prog - warm) / max(1e-8, 1.0 - warm)
            return float(PAIR_USE_PROB_INIT + (PAIR_USE_PROB_FINAL - PAIR_USE_PROB_INIT) * adj_prog)
        except Exception:
            return float(PAIR_USE_PROB_FINAL)
    #get next pair batch with hard mining
    def _next_pair_batch(self, model):
        if self._pair_iter is None:
            self._pair_iter = iter(self.pair_loader)
        try:
            t_more, t_less = next(self._pair_iter)
        except StopIteration:
            self._pair_iter = iter(self.pair_loader)
            t_more, t_less = next(self._pair_iter)
        if not PAIR_HARD_MINING or PAIR_HARD_MULT <= 1:
            return t_more, t_less
        more_list = [t_more]; less_list = [t_less]
        for _ in range(PAIR_HARD_MULT - 1):
            try:
                tm, tl = next(self._pair_iter)
            except StopIteration:
                self._pair_iter = iter(self.pair_loader)
                tm, tl = next(self._pair_iter)
            more_list.append(tm); less_list.append(tl)
        big_more = {k: torch.cat([d[k] for d in more_list], dim=0) for k in more_list[0]}
        big_less = {k: torch.cat([d[k] for d in less_list], dim=0) for k in less_list[0]}
        if PAIR_HARD_WITH_NO_GRAD:
            with torch.no_grad():
                lm = model(**{k: v.to(model.device, non_blocking=True) for k, v in big_more.items()}).logits[:, 1]
                ll = model(**{k: v.to(model.device, non_blocking=True) for k, v in big_less.items()}).logits[:, 1]
        else:
            lm = model(**{k: v.to(model.device, non_blocking=True) for k, v in big_more.items()}).logits[:, 1]
            ll = model(**{k: v.to(model.device, non_blocking=True) for k, v in big_less.items()}).logits[:, 1]
        violation = (ll - lm + self.pair_margin).detach()
        B = t_more["input_ids"].size(0)
        top_idx = torch.topk(violation, k=B, largest=True).indices.cpu()
        select_more = {k: v[top_idx] for k, v in big_more.items()}
        select_less = {k: v[top_idx] for k, v in big_less.items()}
        return select_more, select_less
    #override compute_loss to add class-weighted CE and pairwise hinge
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels") if "labels" in inputs else inputs.get("label")
        outputs = model(**{k: v for k, v in inputs.items() if k in ["input_ids","attention_mask","labels"]})
        logits = outputs.get("logits")

        #simplified robust loss: class-weighted CE actively handles imbalance
        if self.class_weights is not None:
            ce = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            ce = nn.CrossEntropyLoss()
        ce_loss = ce(logits.view(-1, 2), labels.view(-1))

        #pairwise hinge loss from CrowS-Pairs
        pair_loss = torch.tensor(0.0, device=logits.device)
        cur_lambda = self._current_pair_lambda()
        use_prob   = self._current_pair_use_prob()
        if self._pair_iter is not None and cur_lambda > 0.0:
            if random.random() < use_prob:
                t_more, t_less = self._next_pair_batch(model)
                t_more = {k: v.to(model.device, non_blocking=True) for k, v in t_more.items()}
                t_less = {k: v.to(model.device, non_blocking=True) for k, v in t_less.items()}
                logits_more = model(**t_more).logits[:, 1]
                logits_less = model(**t_less).logits[:, 1]
                pairwise = torch.relu(logits_less - logits_more + self.pair_margin).mean()
                pair_loss = cur_lambda * pairwise
        #logging
        try:
            if hasattr(self, "state") and (self.state.global_step % max(1, self.args.logging_steps) == 0):
                self.log({"pair_loss": float(pair_loss.detach().cpu()),
                          "pair_lambda_cur": float(cur_lambda),
                          "pair_use_prob": float(use_prob)})
        except Exception:
            pass
        loss = ce_loss + pair_loss
        return (loss, outputs) if return_outputs else loss

#crowS Pairwise metric evaluation
def evaluate_crows_pairwise(crows_pairs: Optional[Tuple[List[str], List[str]]], tokenizer, model, temperature: Optional[float]=None) -> Optional[float]:
    if not crows_pairs:
        print("[Pairwise] No CrowS pairs; skipping pairwise metric.")
        return None
    s_more, s_less = crows_pairs
    #scoring function
    def score(texts):
        toks = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            logits = model(**{k: v.to(model.device, non_blocking=True) for k, v in toks.items()}).logits
            if CALIBRATE_PAIRWISE_METRIC and (temperature is not None) and np.isfinite(temperature) and temperature > 0:
                logits = logits / float(temperature)
            probs = softmax(logits.detach().cpu().numpy(), axis=1)[:, 1]
        return probs
    #batch scoring to avoid OOM
    def batched(seq, bs=256):
        for i in range(0, len(seq), bs):
            yield seq[i:i+bs]
    p_more, p_less = [], []
    for b_more, b_less in zip(batched(s_more), batched(s_less)):
        p_more.extend(score(b_more))
        p_less.extend(score(b_less))
    p_more = np.array(p_more); p_less = np.array(p_less)
    y = np.concatenate([np.ones_like(p_more), np.zeros_like(p_less)])
    s = np.concatenate([p_more, p_less])
    frac_more_gt_less = float((p_more > p_less).mean())
    frac_less_gt_more = 1.0 - frac_more_gt_less
    #AUROC
    try:
        auroc = roc_auc_score(y, s)
    except Exception:
        auroc = float("nan")
    print(f"[CrowS pairwise] detector fraction P(stereo)>P(anti) = {frac_more_gt_less:.3f}")
    print(f"[CrowS pairwise] LM-preference fraction P(anti)>P(stereo) = {frac_less_gt_more:.3f}")
    print(f"[CrowS pairwise] means: stereo={p_more.mean():.3f}, anti={p_less.mean():.3f}, AUROC={auroc:.3f}")
    return frac_more_gt_less

def main():
    print("=== Loading & converting datasets ===")
    train_ds, eval_ds, crows_pairs = build_combined()

    #rebalance Bios within train (post-split)
    if BIB_REBALANCE_STAGE == "train_only" and BIB_TRAIN_TARGET_POS_FRAC > 0.0:
        train_ds = rebalance_source_to_pos_frac(train_ds, "bios", BIB_TRAIN_TARGET_POS_FRAC, SEED, "undersample_neg")

    #global balance (pre-mix)
    train_ds = rebalance_train(train_ds, TARGET_POS_FRAC, SEED)

    #cap on Bios
    if MAX_BIOS is not None:
        train_ds = cap_bios(train_ds, MAX_BIOS, SEED)

    #mix sources with stratified label sampling
    train_ds = mix_sources(
        train_ds,
        {"crows": 0.25, "stereoset": 0.20, "bios": 0.20, "civil": 0.25, "hatex": 0.10},
        SEED,
    )

    #ensure exact 50/50 after mixing
    train_ds = rebalance_to_pos_frac(train_ds, 0.5, SEED)
    eval_ds = rebalance_eval_per_source(eval_ds, 0.5, SEED)

    USE_CLASS_WEIGHTS = os.getenv("USE_CLASS_WEIGHTS", "1") == "1"
    if USE_CLASS_WEIGHTS:
        labels_np = np.array(train_ds["label"])
        pos = int((labels_np == 1).sum()); neg = int((labels_np == 0).sum()); total = len(labels_np)
        w0 = total / (2.0 * max(neg, 1)); w1 = total / (2.0 * max(pos, 1))
        class_weights = torch.tensor([w0, w1], dtype=torch.float32)
    else:
        class_weights = None


    print("=== Tokenizer & model ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    #pre-train probe on CrowS-Pairs
    if crows_pairs:
        s_more, s_less = crows_pairs
        #tokenize
        def _p1(texts):
            tok = tokenizer(texts[:16], truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt").to(model.device, non_blocking=True)
            with torch.no_grad():
                return torch.softmax(model(**tok).logits, dim=1)[:,1].cpu().numpy()
        #run probe
        try:
            p_more, p_less = _p1(s_more), _p1(s_less)
            print("[Pre-train probe] frac stereo>anti:", float((p_more > p_less).mean()))
        except Exception as e:
            print("[Pre-train probe] skipped due to:", e)

    eval_sources = list(eval_ds["source"])
    #warn if bios positives in eval are very low
    try:
        eval_df = eval_ds.to_pandas()
        bios_eval = eval_df[eval_df["source"] == "bios"]
        bios_pos = int((bios_eval["label"] == 1).sum())
        if bios_pos < 5 and not (EVAL_FORCE_BIOS_POS > 0 or (EVAL_FORCE_BIOS_POS_FRAC and EVAL_FORCE_BIOS_POS_FRAC > 0.0)):
            print(f"[WARN] Bios positives in EVAL are very low ({bios_pos}). "
                  f"Set EVAL_FORCE_BIOS_POS or EVAL_FORCE_BIOS_POS_FRAC to nudge within-Bios split.")
    except Exception:
        pass

    train_tok, eval_tok = tokenize(train_ds, eval_ds, tokenizer)

    print("Build CrowS pairwise mini-dataloader")
    pair_loader = None
    if crows_pairs is not None:
        s_more, s_less = crows_pairs
        pair_dataset = CrowSPairDataset(s_more, s_less)
        collator = CrowsCollator(tokenizer, MAX_LEN)
        pair_loader = DataLoader(
            pair_dataset,
            batch_size=PAIR_BS,
            shuffle=True,
            drop_last=True,
            collate_fn=collator,
            num_workers=0 if IS_WINDOWS else 2,
            pin_memory=True
        )
        print(f"[CrowS pairwise] pairs available: {len(pair_dataset)} (batch size {PAIR_BS})")

    print("Metrics")
    f1_metric = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        micro_f1 = f1_metric.compute(predictions=preds, references=labels)["f1"]
        p, r, f1m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        return {"f1_micro": micro_f1, "f1_macro": f1m, "precision_macro": p, "recall_macro": r}

    print("Training args (GPU-aware)")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        num_train_epochs=EPOCHS,
        logging_steps=50,
        learning_rate=LR,
        save_steps=500,
        save_total_limit=2,
        seed=SEED,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        gradient_accumulation_steps=1,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        weight_decay=0.01,
        warmup_ratio=0.06,
        max_grad_norm=1.0,
    )

    print("Trainer init")
    trainer = WeightedTrainer(
        class_weights=class_weights,
        pair_loader=pair_loader,
        pair_lambda=PAIR_LAMBDA,
        pair_lambda_final=PAIR_LAMBDA_FINAL,
        pair_margin=PAIR_MARGIN,
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        compute_metrics=compute_metrics,
    )

    print("Training")
    trainer.train()

    print("Evaluation (calibrated)")
    eval_res = trainer.evaluate(eval_dataset=eval_tok)
    print("Eval results:", eval_res)

    #temperature scaling (calibration) with safety
    pred = trainer.predict(eval_tok)
    logits = pred.predictions
    y_true = pred.label_ids

    #fit temperature
    T_raw = fit_temperature(logits, y_true)
    T = float(np.clip(T_raw, CAL_T_MIN, CAL_T_MAX))
    probs1 = softmax(logits / T, axis=1)[:, 1]  #calibrated P(biased)

    #if model predicts almost no positives at 0.5, then revert calibration
    pred_rate_05 = float((probs1 >= 0.5).mean())
    if pred_rate_05 < CAL_FAILSAFE_MIN_POS_RATE:
        print(f"[Calibration FAILSAFE] predicted positive rate {pred_rate_05:.4f} < {CAL_FAILSAFE_MIN_POS_RATE:.4f}. Reverting to T=1.0.")
        T = 1.0
        probs1 = softmax(logits, axis=1)[:, 1]
    print(f"[Calibration] learned temperature T_raw={T_raw:.3f} -> clipped T={T:.3f}")

    #reports at deploy threshold (after final probs1)
    print(f"\n=== Evaluation @ DEPLOY_THRESHOLD={DEPLOY_THRESHOLD:.2f} (calibrated) ===")
    y_pred_deploy = (probs1 >= DEPLOY_THRESHOLD).astype(int)
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred_deploy))
    print(classification_report(y_true, y_pred_deploy, digits=4))
    print("Positive rate predicted:", float((y_pred_deploy==1).mean()))

    #reports at 0.5 (calibrated)
    print("\n=== Evaluation @ 0.5 (calibrated) ===")
    y_pred_05 = (probs1 >= 0.5).astype(int)
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred_05))
    print(classification_report(y_true, y_pred_05, digits=4))
    print("Positive rate predicted:", float((y_pred_05==1).mean()))

    #per-source breakdown at 0.5
    src_arr = np.array(eval_sources)
    for src in ["crows", "stereoset", "bios"]:
        mask = (src_arr == src)
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred_05[mask]
        print(f"\n--- Per-source: {src} @ 0.5 (calibrated) ---")
        print("Confusion matrix:\n", confusion_matrix(yt, yp))
        print(classification_report(yt, yp, digits=4))
        print("Macro-F1:", f1_score(yt, yp, average="macro", zero_division=0))

    #threshold tuning for macro-F1 (calibrated)
    print("\n=== Threshold tuning (macro-F1, calibrated) ===")
    ths = np.linspace(0.05, 0.95, 19)
    best_f1, best_th = -1.0, 0.5
    for th in ths:
        y_hat = (probs1 >= th).astype(int)
        f1m = f1_score(y_true, y_hat, average="macro", zero_division=0)
        if f1m > best_f1:
            best_f1, best_th = f1m, th
    print(f"[Threshold tuning] best_macro_F1={best_f1:.4f} at threshold={best_th:.2f}")

    #per-source at best threshold (calibrated)
    per_src_thresholds = {}
    for src in ["crows", "stereoset", "bios"]:
        mask = (src_arr == src)
        if mask.sum() == 0:
            continue
        y_hat = (probs1[mask] >= best_th).astype(int)
        f1m = f1_score(y_true[mask], y_hat, average="macro", zero_division=0)
        per_src_thresholds[src] = {"best_macro_f1": float(f1m), "best_threshold": float(best_th)}
        print(f"[{src}] macro-F1 @ {best_th:.2f} = {f1m:.4f}")

    #global tuned threshold summary
    print("\n=== Confusion & reports @ tuned global threshold (calibrated) ===")
    y_hat_global = (probs1 >= best_th).astype(int)
    print("Confusion matrix (global tuned):\n", confusion_matrix(y_true, y_hat_global))
    print(classification_report(y_true, y_hat_global, digits=4))
    for src in ["crows", "stereoset", "bios"]:
        mask = (src_arr == src)
        if mask.sum() == 0:
            continue
        print(f"\n--- Per-source: {src} @ tuned {best_th:.2f} (calibrated) ---")
        print("Confusion matrix:\n", confusion_matrix(y_true[mask], y_hat_global[mask]))
        print(classification_report(y_true[mask], y_hat_global[mask], digits=4))

    #crowS pairwise metric
    _ = evaluate_crows_pairwise(crows_pairs, tokenizer, model, temperature=T)

    print("=== Saving model ===")
    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Saved model and tokenizer to {SAVE_DIR}")

    #export calibration and thresholds
    export = {
        "temperature": float(T),
        "deploy_threshold": float(DEPLOY_THRESHOLD),
        "global_threshold_tuned": float(best_th),
        "per_source_thresholds": per_src_thresholds,
        "pairwise": {
            "lambda_init": PAIR_LAMBDA,
            "lambda_final": PAIR_LAMBDA_FINAL,
            "margin": PAIR_MARGIN,
            "use_prob_init": PAIR_USE_PROB_INIT,
            "use_prob_final": PAIR_USE_PROB_FINAL,
            "hard_mining": bool(PAIR_HARD_MINING),
            "hard_mult": int(PAIR_HARD_MULT)
        },
        "bios_eval_nudge": {
            "min_pos": int(EVAL_FORCE_BIOS_POS),
            "min_pos_frac": float(EVAL_FORCE_BIOS_POS_FRAC)
        },
        "calibrate_pairwise_metric": bool(CALIBRATE_PAIRWISE_METRIC),
    }
    #write JSON
    with open(os.path.join(SAVE_DIR, "calibration_and_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)
    print(f"[Export] wrote {os.path.join(SAVE_DIR,'calibration_and_thresholds.json')}")
    print(f"Deploy threshold (env): {DEPLOY_THRESHOLD:.2f}")
    print(f"Suggested deploy threshold (macro-F1-tuned, calibrated): {best_th:.2f}")

if __name__ == "__main__":
    main()