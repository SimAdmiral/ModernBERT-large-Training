# ModernBERT CTI Pipeline: Entity Annotation & Tokenization

## Overview
This document outlines how we process Cyber Threat Intelligence (CTI) text for the **ModernBERT-large** model. The goal is to take raw text labeled by humans (character spans) and convert it into a high-precision, token-level dataset ready for training.

We prioritize three things in this pipeline:
1.  **Exact alignment:** Ensuring human highlights match the model's tokens perfectly.
2.  **Complex Entity Support:** Handling tricky CTI entities like long URLs, file paths, and multi-word threat group names.
3.  **BIOUL Tagging:** Using a granular tagging scheme to improve model performance.

---

## 1. Input Data: The Human Layer
Humans don't think in tokens; they think in phrases. Therefore, our input data relies on **character offsets**. The annotator simply highlights the start and end of an entity in the original string.

**The JSON Schema:**
```json
{
  "id": "sample_001",
  "text": "APT28 used Spear Phishing to deploy Emotet via https://malware.site/download",
  "entities": [
    { "start": 0, "end": 5, "label": "CTI_GROUP" },
    { "start": 11, "end": 26, "label": "MITRE_TECHNIQUE" },
    { "start": 37, "end": 43, "label": "MALWARE" },
    { "start": 48, "end": 84, "label": "URL" }
  ]
}
```
*   **Note:** Offsets are character-based. The `end` index is exclusive. We allow multiple entities per sentence but do not allow overlapping entities.

---

## 2. Tokenization Logic (ModernBERT)
We use the `answerdotai/ModernBERT-large` tokenizer. This is a Byte-level BPE tokenizer (similar to RoBERTa), which is space-sensitive and offset-aware.

### The Special Character: `Ġ`
You will see the `Ġ` symbol frequently in the output (e.g., `Ġused`).
*   This represents a **leading whitespace**.
*   **Do not remove it.** It is required to preserve exact character positions and ensure the model understands word boundaries.

### Offset Mapping
To align the human "spans" with the machine "tokens," we run the tokenizer with `return_offsets_mapping=True`. This generates a start/end character position for every single subword token, which we cross-reference against the input entities.

---

## 3. Labeling Strategy: BIOUL
We use **BIOUL** rather than the standard BIO format. This adds specific tags for single-unit entities and the last token of a phrase, which helps Transformer models converge faster on complex boundaries.

| Tag | Meaning | Use Case |
| :--- | :--- | :--- |
| **B-** | **Begin** | The first token of a multi-token entity. |
| **I-** | **Inside** | Any token in the middle of an entity. |
| **L-** | **Last** | The final token of a multi-token entity. |
| **U-** | **Unit** | An entity that consists of only one token. |
| **O** | **Outside** | Text that is not part of an entity. |

### Example Scenario
**Entity:** `Spear Phishing` (Label: `MITRE_TECHNIQUE`)
**Tokens:** `Spe`, `ar`, `Ph`, `ishing`

**Correct Labeling:**
*   `Spe` $\rightarrow$ **B**-MITRE_TECHNIQUE
*   `ar` $\rightarrow$ **I**-MITRE_TECHNIQUE
*   `Ph` $\rightarrow$ **I**-MITRE_TECHNIQUE
*   `ishing` $\rightarrow$ **L**-MITRE_TECHNIQUE

---

## 4. Handling Edge Cases: URLs
URLs are common in CTI but difficult for tokenizers because they are semantically one "object" but syntactically a mess of punctuation and words.

**Example:** `https://malware.site/download`
The tokenizer splits this into roughly 8 parts (`https`, `://`, `mal`, `ware`, etc.).

Using the logic `token_start < entity_end AND token_end > entity_start`:
1.  We capture **all** sub-tokens associated with the URL.
2.  We apply BIOUL to the sequence (Start with **B**, fill with **I**, end with **L**).
3.  This ensures the model learns the entire string is a single `URL` entity.

---

## 5. Final Output Format
The pipeline produces a JSON object fully compatible with the Hugging Face `Trainer`.

```json
{
  "tokens": [
    "AP", "T", "28",
    "Ġused",
    "ĠSpe", "ar", "ĠPh", "ishing",
    "Ġto", "Ġdeploy",
    "ĠEm", "ot", "et",
    "Ġvia",
    "Ġhttps", "://", "mal", "ware", ".", "site", "/", "download"
  ],
  "ner_tags": [
    "B-CTI_GROUP", "I-CTI_GROUP", "L-CTI_GROUP",
    "O",
    "B-MITRE_TECHNIQUE", "I-MITRE_TECHNIQUE", "I-MITRE_TECHNIQUE", "L-MITRE_TECHNIQUE",
    "O", "O",
    "B-MALWARE", "I-MALWARE", "L-MALWARE",
    "O",
    "B-URL", "I-URL", "I-URL", "I-URL", "I-URL", "I-URL", "I-URL", "L-URL"
  ]
}
```

## 6. Validation & Quality Assurance
Before training, every sample undergoes an automated logic check:
*   **Sequence Integrity:** Ensures no `I-` or `L-` tags appear without a preceding `B-`.
*   **Closure:** Ensures all entities are properly "closed" (ending in `L` or `U`).
*   **Gap Check:** Ensures no `O` tags accidentally appear inside an open entity.

Invalid samples are automatically rejected to prevent model hallucinations during training.