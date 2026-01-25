# !pip install transformers torch
import json
from transformers import AutoTokenizer

MODEL = "answerdotai/ModernBERT-large"

ENTITIES = [
    "URL", "MALWARE", "MITRE_TACTIC", "MITRE_TECHNIQUE",
    "CTI_GROUP", "CTI_CAMPAIGN", "TOOL", "DOMAIN"
]

def bioul_tags(label, length):
    """Generate BIOUL tags for a given entity length."""
    if length == 1:
        return [f"U-{label}"]
    return [f"B-{label}"] + [f"I-{label}"] * (length - 2) + [f"L-{label}"]

def load_data(path):
    """Load JSON dataset with text and entity spans."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def align_tokens_to_entities(offsets, entities):
    """Return a list of BIOUL tags aligned to tokens."""
    ner_tags = ["O"] * len(offsets)

    for ent in entities:
        start_char = ent["start"]
        end_char = ent["end"]
        label = ent["label"]

        # Find tokens that overlap with entity span
        token_indices = [
            i for i, (token_start, token_end) in enumerate(offsets)
            if token_start < end_char and token_end > start_char
        ]

        if not token_indices:
            print(f"Warning: entity '{ent['label']}' [{start_char}:{end_char}] not aligned to any token")
            continue

        tags = bioul_tags(label, len(token_indices))
        for idx, tag in zip(token_indices, tags):
            ner_tags[idx] = tag

    return ner_tags

def convert_example(example, tokenizer):
    text = example["text"]
    entities = example.get("entities", [])

    # Tokenize with offsets
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]

    ner_tags = align_tokens_to_entities(offsets, entities)

    return {
        "tokens": tokens,
        "ner_tags": ner_tags
    }

def convert_dataset(input_path, output_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    raw_data = load_data(input_path)

    dataset = [convert_example(example, tokenizer) for example in raw_data]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(dataset)} samples to {output_path}")

if __name__ == "__main__":
    convert_dataset("input_spans.json", "output_ner_dataset.json")
