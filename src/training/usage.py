import json
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Path to your trained model folder
MODEL_PATH = "./modernbert-cti-bioul"

def clean_entity_result(raw_results):
    """
    Hugging Face pipelines with BIOUL tags can sometimes output fragmented entities.
    This helper merges subwords and handles the B/I/L/U logic to return clean spans.
    """
    merged_entities = []
    current_entity = None

    for item in raw_results:
        word = item['word']
        tag = item['entity_group'] # aggregation_strategy="simple" groups valid spans
        score = item['score']
        start = item['start']
        end = item['end']

        # Clean up ModernBERT tokenizer artifacts (e.g. "Ġ" for space)
        clean_word = word.replace('Ġ', ' ').strip()

        # If it's a new group provided by the pipeline aggregator
        if current_entity is None:
            current_entity = {
                "label": tag,
                "text": clean_word,
                "score": float(score),
                "start": start,
                "end": end
            }
        elif tag == current_entity["label"] and start == current_entity["end"]:
            # Direct adjacency (subword merge)
            current_entity["text"] += word.replace('Ġ', '') # Append without space if subword
            current_entity["end"] = end
            current_entity["score"] = (current_entity["score"] + score) / 2
        elif tag == current_entity["label"] and start == current_entity["end"] + 1:
            # Space adjacency
            current_entity["text"] += " " + clean_word
            current_entity["end"] = end
            current_entity["score"] = (current_entity["score"] + score) / 2
        else:
            # Flush current and start new
            merged_entities.append(current_entity)
            current_entity = {
                "label": tag,
                "text": clean_word,
                "score": float(score),
                "start": start,
                "end": end
            }
            
    if current_entity:
        merged_entities.append(current_entity)

    return merged_entities

def main():
    print(f"Loading model from {MODEL_PATH}...")
    
    # 1. Load Model & Tokenizer
    # We use the pipeline API which handles tokenization and label mapping automatically
    try:
        nlp = pipeline(
            "token-classification", 
            model=MODEL_PATH, 
            tokenizer=MODEL_PATH, 
            aggregation_strategy="simple", # Tries to merge B-I-L tags automatically
            device=0 # Change to -1 for CPU
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did you run the training script and save to ./modernbert-cti-bioul?")
        return

    # 2. Define CTI Text
    text = (
        "Analysts observed APT29 utilizing a new variant of Cobalt Strike "
        "served from https://update-kernel-linux.org/main.exe to target "
        "energy sector domains."
    )

    print(f"\nInput Text:\n{text}\n")

    # 3. Run Inference
    raw_results = nlp(text)

    # 4. Process and Print
    print("-" * 60)
    print(f"{'ENTITY':<30} | {'LABEL':<20} | {'CONFIDENCE':<10}")
    print("-" * 60)

    for ent in raw_results:
        # Pipeline handles "Ġ" cleanup mostly, but we ensure display is clean
        entity_text = ent['word'].strip()
        label = ent['entity_group']
        score = ent['score']
        
        print(f"{entity_text:<30} | {label:<20} | {score:.4f}")

    print("-" * 60)

    # 5. Output as JSON
    output_json = []
    for ent in raw_results:
        output_json.append({
            "text": ent['word'].strip(),
            "label": ent['entity_group'],
            "score": float(ent['score']),
            "start": ent['start'],
            "end": ent['end']
        })
        
    print("\nJSON Output:")
    print(json.dumps(output_json, indent=2))

if __name__ == "__main__":
    main()