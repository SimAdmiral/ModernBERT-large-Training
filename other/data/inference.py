from transformers import pipeline

# 1. Load the pipeline
ner_pipeline = pipeline(
    "token-classification",
    model="./final_cti_model_complete", 
    tokenizer="./final_cti_model_complete",
    aggregation_strategy="simple" 
)

text = "APT28 used Emotet malware and a C2 server at malicious-c2.net to perform Credential Access."

entities = ner_pipeline(text)

print(f"{'Entity':<25} | {'Label':<20} | {'Score':<10}")
print("-" * 60)

for entity in entities:
    print(f"{entity['word']:<25} | {entity['entity_group']:<20} | {entity['score']:.4f}")