import re
import random
import json
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from lxml import etree

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers

import torch

OUT_DIR = os.path.join(os.path.dirname(__file__), "svg_data_part4")
os.makedirs(OUT_DIR, exist_ok=True)
print("Saving outputs to:", OUT_DIR)

icons = load_dataset("starvector/svg-icons-simple", split="train")
emoji = load_dataset("starvector/svg-emoji-simple", split="train")
stack = load_dataset("starvector/svg-stack-simple", split="train")

all_data = list(icons) + list(emoji) + list(stack)
print("Total raw samples:", len(all_data))

def clean_svg(svg: str) -> str:
    svg = re.sub(r'<!--.*?-->', '', svg, flags=re.DOTALL)

    if "<script" in svg or "<style" in svg:
        return None

    if svg.count("<svg") > 1:
        return None

    svg = re.sub(r'\s+', ' ', svg).strip()

    svg = re.sub(r'(\d+\.\d+)', lambda m: f"{float(m.group()):.2f}", svg)

    svg = svg.replace('"', "'")

    svg = re.sub(r"=\s*''", "='0'", svg)

    svg = re.sub(r"=\s*none", "='none'", svg)

    if not svg.strip().endswith("</svg>"):
        svg += "</svg>"

    return svg

def is_valid_xml(svg: str) -> bool:
    try:
        etree.fromstring(svg.encode("utf-8"))
        return True
    except Exception:
        return False

MIN_LEN = 50
MAX_LEN = 2000

def length_ok(svg: str) -> bool:
    return MIN_LEN < len(svg) < MAX_LEN

cleaned_svgs = []
for item in tqdm(all_data, desc="Cleaning / filtering"):
    svg = item["Svg"]
    svg = clean_svg(svg)

    if svg is None:
        continue
    if not length_ok(svg):
        continue
    if not is_valid_xml(svg):
        continue

    cleaned_svgs.append(svg)

print("Final cleaned dataset size:", len(cleaned_svgs))

lengths = [len(s) for s in cleaned_svgs]
plt.hist(lengths, bins=50)
plt.title("SVG Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "length_hist.png"), dpi=150)
plt.close()
print("Saved length histogram to length_hist.png")

random.shuffle(cleaned_svgs)
n = len(cleaned_svgs)

train_svgs = cleaned_svgs[: int(0.98 * n)]
val_svgs   = cleaned_svgs[int(0.98 * n): int(0.99 * n)]
test_svgs  = cleaned_svgs[int(0.99 * n):]

print("Split sizes:", len(train_svgs), len(val_svgs), len(test_svgs))

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = BpeTrainer(
    vocab_size=4000,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
)

tokenizer.train_from_iterator(train_svgs, trainer)
print("Tokenizer vocab size:", tokenizer.get_vocab_size())

MAX_TOK_LEN = 2048

def encode_list(svg_list):
    encoded = []
    for svg in tqdm(svg_list, desc="Encoding"):
        ids = tokenizer.encode(svg).ids
        if len(ids) <= MAX_TOK_LEN:
            encoded.append(ids)
    return encoded

train_tokens = encode_list(train_svgs)
val_tokens   = encode_list(val_svgs)
test_tokens  = encode_list(test_svgs)

train_lens = [len(x) for x in train_tokens]
val_lens   = [len(x) for x in val_tokens]
test_lens  = [len(x) for x in test_tokens]

train_total = sum(train_lens)
val_total   = sum(val_lens)
test_total  = sum(test_lens)

print("Train tokens:", train_total)
print("Val tokens:",   val_total)
print("Test tokens:",  test_total)
print("TOTAL TOKENS:", train_total + val_total + test_total)

tok_path   = os.path.join(OUT_DIR, "svg_tokenizer.json")
train_path = os.path.join(OUT_DIR, "train_tokens.pt")
val_path   = os.path.join(OUT_DIR, "val_tokens.pt")
test_path  = os.path.join(OUT_DIR, "test_tokens.pt")
meta_path  = os.path.join(OUT_DIR, "metadata.json")

tokenizer.save(tok_path)
torch.save(train_tokens, train_path)
torch.save(val_tokens,   val_path)
torch.save(test_tokens,  test_path)

metadata = {
    "vocab_size": tokenizer.get_vocab_size(),
    "train_samples": len(train_tokens),
    "val_samples": len(val_tokens),
    "test_samples": len(test_tokens),
    "train_tokens": train_total,
    "val_tokens": val_total,
    "test_tokens": test_total,
    "max_len": MAX_TOK_LEN,
}

with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("Tokenizer   :", tok_path)
print("Train tokens:", train_path)
print("Val tokens  :", val_path)
print("Test tokens :", test_path)
print("Metadata    :", meta_path)
