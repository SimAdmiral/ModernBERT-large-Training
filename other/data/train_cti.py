import json
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate
from seqeval.metrics import classification_report

# 1. SETUP LABELS
# ==============================================================================
ENTITIES = [
    "URL", "MALWARE", "MITRE_TACTIC", "MITRE_TECHNIQUE", 
    "CTI_GROUP", "CTI_CAMPAIGN", "TOOL", "DOMAIN"
]

TAGS = ["O"]
for ent in ENTITIES:
    TAGS.extend([f"B-{ent}", f"I-{ent}", f"L-{ent}", f"U-{ent}"])

label2id = {tag: i for i, tag in enumerate(TAGS)}
id2label = {i: tag for i, tag in enumerate(TAGS)}

print(f"Training with {len(TAGS)} tags.")

# 2. LOAD DATASET
# ==============================================================================
def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

# Ensure these files exist in your folder
raw_datasets = {
    "train": load_json_data("train.json"),
    "validation": load_json_data("val.json"),
    "test": load_json_data("test.json")
}

# 3. TOKENIZATION
# ==============================================================================
model_checkpoint = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        max_length=1024 
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                try:
                    label_ids.append(label2id[label[word_idx]])
                except KeyError:
                    print(f"Error: Label '{label[word_idx]}' in dataset not found in entity list.")
                    raise
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("Tokenizing...")
tokenized_datasets = {
    x: raw_datasets[x].map(tokenize_and_align_labels, batched=True) 
    for x in raw_datasets
}

# 4. MODEL & METRICS
# ==============================================================================
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(TAGS),
    id2label=id2label,
    label2id=label2id
)

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [TAGS[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [TAGS[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 5. TRAINING
# ==============================================================================
args = TrainingArguments(
    output_dir="cti-modernbert-final",
    eval_strategy="epoch",            # Updated from evaluation_strategy (deprecated)
    save_strategy="epoch",
    learning_rate=2e-5,
    
    # ADJUSTED FOR MEMORY:
    per_device_train_batch_size=4,    # Lowered from 8 to prevent OOM
    gradient_accumulation_steps=2,    # Simulates batch size of 8 (4 * 2)
    per_device_eval_batch_size=4,
    
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    report_to="none",
    fp16=True,                        # ModernBERT works great with Mixed Precision
    
    # Gradient Checkpointing saves massive memory for Large models
    gradient_checkpointing=True       
)


data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# 6. SAVE & EVALUATE
# ==============================================================================
print("\n--- Final Evaluation on Test Set ---")
metrics = trainer.evaluate(tokenized_datasets["test"])
print("Global Metrics:", metrics)

predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(predictions, axis=2)

true_predictions = [
    [TAGS[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [TAGS[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

print("\nDetailed Per-Entity Report:")
print(classification_report(true_labels, true_predictions))

# Save
trainer.save_model("final_cti_model_complete")
tokenizer.save_pretrained("final_cti_model_complete")
print("Model saved to ./final_cti_model_complete")