import json
import google.generativeai as genai

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"

# The specific CTI labels to extract
ALLOWED_LABELS = [
    "CTI_GROUP", "CTI_CAMPAIGN", "MALWARE", "MITRE_TECHNIQUE", 
    "MITRE_TACTIC", "TOOL", "DOMAIN", "URL"
]

def calculate_offsets(sentence, entity_text, label):
    """
    Finds the exact start/end indices of entity_text within sentence.
    """
    start = sentence.find(entity_text)
    
    # Validation: Entity must exist in the sentence
    if start == -1:
        return None
    
    return {
        "start": start,
        "end": start + len(entity_text),
        "label": label
    }

def analyze_raw_text(raw_text):
    """Sends raw text to Google Gemini to extract CTI data."""
    
    genai.configure(api_key=GOOGLE_API_KEY)
    # Gemini 1.5 Flash is fast and good for extraction tasks
    model = genai.GenerativeModel('gemini-1.5-flash') 

    prompt = f"""
    You are a CTI Data Annotator.
    
    Task:
    1. Read the text below.
    2. Extract sentences that contain CTI entities.
    3. Identify entities using ONLY these labels: {", ".join(ALLOWED_LABELS)}.
    
    CRITICAL RULES:
    - Return a JSON List.
    - The "text" field must be an EXACT COPY of the sentence from the input. Do not paraphrase.
    - The entity "text" must be an EXACT COPY of the substring found in that sentence.
    
    Input Text:
    {raw_text}
    
    Output JSON Format:
    [
      {{
        "text": "Exact sentence string.",
        "entities": [
          {{ "text": "substring", "label": "LABEL_NAME" }}
        ]
      }}
    ]
    """

    print("Querying Google Gemini...")
    try:
        response = model.generate_content(prompt)
        # Clean potential markdown wrapping
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        print(f"Error calling API or parsing JSON: {e}")
        return []

def main():
    # ---------------------------------------------------------
    # INPUT RAW TEXT HERE
    # ---------------------------------------------------------
    raw_text_input = """
    In recent campaigns, APT29 has shifted tactics to focus on cloud environments.
    They utilized a new variant of Duke malware to maintain persistence.
    The group was observed extracting data via https://cloud-upload-secure.com/api.
    Additionally, Fin7 targeted retail sectors using point-of-sale malware.
    """

    # 1. Get AI Analysis
    ai_results = analyze_raw_text(raw_text_input)

    final_dataset = []

    # 2. Post-Process: Calculate offsets for training
    print("Aligning entities to character offsets...")
    
    for item in ai_results:
        sentence = item.get("text", "").strip()
        raw_entities = item.get("entities", [])
        
        valid_entities = []
        
        for ent in raw_entities:
            ent_text = ent.get("text")
            label = ent.get("label")
            
            # Skip invalid labels
            if label not in ALLOWED_LABELS:
                continue

            # Calculate mathematical offsets
            offset_data = calculate_offsets(sentence, ent_text, label)
            
            if offset_data:
                valid_entities.append(offset_data)
            else:
                print(f"Warning: Gemini hallucinated text '{ent_text}' not found in sentence.")

        if valid_entities:
            final_dataset.append({
                "text": sentence,
                "entities": valid_entities
            })

    # 3. Output
    output_filename = "input_spans.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
    
    print("-" * 50)
    print(json.dumps(final_dataset, indent=2))
    print("-" * 50)
    print(f"Saved to {output_filename}")

if __name__ == "__main__":
    main()