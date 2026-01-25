import json
from transformers import pipeline

# 1. Load the Pipeline
# aggregation_strategy="simple" automatically merges B-I-L-U tags into words
nlp = pipeline(
    "token-classification", 
    model="./modernbert-cti-bioul", 
    tokenizer="./modernbert-cti-bioul", 
    aggregation_strategy="simple" 
)

# 2. Input Text
text = (
    "Analysts observed APT29 utilizing a new variant of Cobalt Strike "
    "served from https://update-kernel-linux.org/main.exe to target "
    "energy sector domains."
)

# 3. Run Inference
results = nlp(text)

# 4. Print Clean Table
print(f"{'ENTITY':<35} | {'LABEL':<20} | {'SCORE':<5}")
print("-" * 70)

formatted_data = []

for item in results:
    # Clean up tokenizer artifacts (whitespace)
    clean_text = item['word'].strip()
    
    print(f"{clean_text:<35} | {item['entity_group']:<20} | {item['score']:.4f}")
    
    # Prepare JSON structure
    formatted_data.append({
        "text": clean_text,
        "label": item['entity_group'],
        "confidence": float(item['score']),
        "start": item['start'],
        "end": item['end']
    })

# 5. Print JSON
print("\nJSON Output:")
print(json.dumps(formatted_data, indent=2))