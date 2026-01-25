# CTI NER Training Documentation (ModernBERT)

## 1. Project Overview
**Goal:** Fine-tune `answerdotai/ModernBERT-large` to recognize specific Cyber Threat Intelligence entities using the **BIOUL** tagging scheme.

**Model:** [answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large)  
**Task:** Token Classification (Named Entity Recognition)  
**Format:** Token-level JSON  

### Target Entities
Based on your requirements, the model will recognize the following 8 classes:

| Entity | Description & Examples |
| :--- | :--- |
| **URL** | Full URLs including paths/params. <br>*(e.g., `http://bad.com/update.exe`)* |
| **MALWARE** | Software designed for malicious purposes. <br>*(e.g., `Emotet`, `Cobalt Strike`, `PlugX`)* |
| **MITRE_TACTIC** | High-level attacker goals. <br>*(e.g., `Credential Access`, `Persistence`)* |
| **MITRE_TECHNIQUE** | Specific methods attackers use. <br>*(e.g., `Spear Phishing`, `T1059`)* |
| **CTI_GROUP** | Hacker/threat actor groups. <br>*(e.g., `APT28`, `Lazarus`, `FIN7`)* |
| **CTI_CAMPAIGN** | Specific operations. <br>*(e.g., `Operation Aurora`, `Sandworm Campaign`)* |
| **TOOL** | Dual-use software (admin/red-team/adversary). <br>*(e.g., `PsExec`, `Mimikatz`, `Metasploit`, `net.exe`)* |
| **DOMAIN** | FQDNs or subdomains without protocol/path. <br>*(e.g., `malicious.badsite.net`, `google.com`)* |

---

## 2. Installation & Environment

```bash
python -m venv venv
venv\Scripts\activate

pip install --upgrade transformers torch datasets seqeval evaluate accelerate scikit-learn
```

---

## 3. Label Mapping (BIOUL)

The **BIOUL** scheme creates 4 tags for every entity plus the "Outside" tag.
*   **Total Labels:** 33 (8 Entities × 4 tags + 1 `O` tag).

### The ID Map

| ID | Label | ID | Label | ID | Label |
|:---|:---|:---|:---|:---|:---|
| 0 | **O** | 11 | L-MITRE_TACTIC | 22 | I-CTI_CAMPAIGN |
| 1 | B-URL | 12 | U-MITRE_TACTIC | 23 | L-CTI_CAMPAIGN |
| 2 | I-URL | 13 | B-MITRE_TECHNIQUE | 24 | U-CTI_CAMPAIGN |
| 3 | L-URL | 14 | I-MITRE_TECHNIQUE | 25 | B-TOOL |
| 4 | U-URL | 15 | L-MITRE_TECHNIQUE | 26 | I-TOOL |
| 5 | B-MALWARE | 16 | U-MITRE_TECHNIQUE | 27 | L-TOOL |
| 6 | I-MALWARE | 17 | B-CTI_GROUP | 28 | U-TOOL |
| 7 | L-MALWARE | 18 | I-CTI_GROUP | 29 | B-DOMAIN |
| 8 | U-MALWARE | 19 | L-CTI_GROUP | 30 | I-DOMAIN |
| 9 | B-MITRE_TACTIC | 20 | U-CTI_GROUP | 31 | L-DOMAIN |
| 10 | I-MITRE_TACTIC | 21 | B-CTI_CAMPAIGN | 32 | U-DOMAIN |

---





## Dataset

For training, we use 3 types of datasets:  
- Training  
- Validation  
- Test  

We use JSON label format, which is human-readable.

### 1️⃣ What your dataset looks like (raw)

```json
{
  "tokens": ["Charming", "Kitten", "The", "by", "was", "FIN7", "and", "in", "detected"],
  "ner_tags": ["B-CTI_GROUP", "L-CTI_GROUP", "O", "O", "O", "U-CTI_GROUP", "O", "O", "O"]
}
```


They are automaticaly convert with Hugging Face’s token classification pipeline handles that automatically. Let me clarify.



## 4. Training Pipeline (Script)


**Run the training:**
```bash
python train_cti.py
```

---

## 5. Inference (How to use it)

Once training is complete, use this script to test the model. The pipeline automatically reconstructs the BIOUL tags into full entities.

**Run the training:** `inference.py`


### Expected Output Example
```text
Entity                    | Label                | Score     
------------------------------------------------------------
Operation Aurora          | CTI_CAMPAIGN         | 0.9912
APT1                      | CTI_GROUP            | 0.9985
Hydraq                    | MALWARE              | 0.9850
Spear Phishing            | MITRE_TECHNIQUE      | 0.9720
PsExec                    | TOOL                 | 0.9940
update.driver-soft.com    | DOMAIN               | 0.9899
```
