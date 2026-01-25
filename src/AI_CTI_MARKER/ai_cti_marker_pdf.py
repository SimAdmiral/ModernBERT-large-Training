import os
import json
import re
import google.generativeai as genai
import pypdf

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Replace with your actual Google API Key
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"

# The specific CTI labels you want to extract
ALLOWED_LABELS = [
    "CTI_GROUP", "CTI_CAMPAIGN", "MALWARE", "MITRE_TECHNIQUE", 
    "MITRE_TACTIC", "TOOL", "DOMAIN", "URL"
]

def extract_text_from_pdf(pdf_path):
    """Reads raw text from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def calculate_offsets(sentence, entity_text, label):
    """
    Finds the exact start/end indices of entity_text within sentence.
    Returns None if text not found exactly.
    """
    start = sentence.find(entity_text)
    if start == -1:
        return None
    
    return {
        "start": start,
        "end": start + len(entity_text),
        "label": label
    }

def analyze_with_gemini(pdf_text):
    """Sends text to Google Gemini to extract CTI data."""
    
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro-latest') # Pro handles large PDFs better

    # Strict prompt to force JSON output
    prompt = f"""
    You are a Cyber Threat Intelligence (CTI) Expert.
    
    Task:
    1. Analyze the text provided below.
    2. Extract sentences that contain CTI entities.
    3. Identify specific entities using ONLY these labels: {", ".join(ALLOWED_LABELS)}.
    4. Return the output as a strict JSON list.

    Rules:
    - COPY the sentence text EXACTLY as it appears in the source. Do not summarize.
    - COPY the entity text EXACTLY as it appears in the sentence.
    - If a sentence has no relevant entities, skip it.
    
    Output Format (JSON List):
    [
      {{
        "text": "Exact sentence from PDF.",
        "entities": [
          {{ "text": "extracted entity substring", "label": "LABEL_NAME" }}
        ]
      }}
    ]

    Input Text:
    {pdf_text}
    """

    print("Sending request to Google Gemini (this may take a moment)...")
    response = model.generate_content(prompt)
    
    # Clean response (remove markdown code blocks if Gemini adds them)
    raw_json = response.text.replace("```json", "").replace("```", "").strip()
    
    try:
        data = json.loads(raw_json)
        return data
    except json.JSONDecodeError:
        print("Error: Gemini did not return valid JSON. Raw response:")
        print(raw_json)
        return []

def process_pdf_to_training_data(pdf_path, output_json_path):
    # 1. Extract Text
    print(f"Extracting text from {pdf_path}...")
    pdf_text = extract_text_from_pdf(pdf_path)

    # 2. Get AI Analysis
    ai_results = analyze_with_gemini(pdf_text)

    final_dataset = []

    # 3. Post-Process: Calculate offsets for training script
    print("Calculating offsets and validating spans...")
    for item in ai_results:
        sentence = item.get("text", "").strip()
        raw_entities = item.get("entities", [])
        
        valid_entities = []
        
        for ent in raw_entities:
            ent_text = ent.get("text")
            label = ent.get("label")
            
            # Verify label is allowed
            if label not in ALLOWED_LABELS:
                continue

            # Calculate mathematical offsets
            offset_data = calculate_offsets(sentence, ent_text, label)
            
            if offset_data:
                valid_entities.append(offset_data)
            else:
                print(f"Warning: Could not find '{ent_text}' inside sentence: '{sentence[:30]}...'")

        # Only add if valid entities were found
        if valid_entities:
            final_dataset.append({
                "text": sentence,
                "entities": valid_entities
            })

    # 4. Save
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Success! Saved {len(final_dataset)} labeled samples to {output_json_path}")
    print("You can now feed this file into your training script.")

if __name__ == "__main__":
    # CHANGE THIS to your PDF file path
    PDF_FILE = "report.pdf" 
    OUTPUT_FILE = "input_spans.json"
    
    if not os.path.exists(PDF_FILE):
        print(f"Please place a '{PDF_FILE}' file in this folder or update the path.")
    else:
        process_pdf_to_training_data(PDF_FILE, OUTPUT_FILE)