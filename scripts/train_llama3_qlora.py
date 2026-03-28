"""
llama3_qlora_finetune.py

Implementation of the lightweight prompt fine-tuning pipeline using LLaMA 3 (8B) 
and QLoRA (Quantized Low-Rank Adaptation) for receipt structured data extraction.

Prerequisites:
    pip install torch transformers peft trl datasets bitsandbytes accelerate
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "./models/llama3-receipt-extractor"
DATASET_PATH = "./data/receipts_sft.jsonl" # Path to your JSONL dataset

# Hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4


# ─── 1. Prompt Formatting Engine ─────────────────────────────────────────────
def format_instruction_prompt(example):
    """
    Format the dataset row into the exact instruction structure required.
    Expects 'ocr_text' and 'json_output' keys in the dataset.
    """
    prompt = f"""### Instruction:
Extract structured information from the receipt text.

### Input:
{example['ocr_text']}

### Output:
{example['json_output']}
<|eot_id|>""" # Using LLaMA 3's end-of-turn token
    
    return {"text": prompt}


def main():
    print("🚀 Initializing LLaMA 3 QLoRA Fine-tuning Pipeline...")
    
    # ─── 2. Quantization Context Setup (QLoRA) ──────────────────────────────
    # Load the massive 8B model natively in 4-bit precision to fit on consumer GPUs
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",       # Normal Float 4 quantization
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Saves memory
    )

    print(f"Loading base model [{MODEL_NAME}] in 4-bit precision...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto" # Auto-distributes weights across available GPUs
    )

    # ─── 3. Parameter-Efficient Fine-Tuning Setup ────────────────────────────
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ] # Targeting all linear layers captures maximum representation capability
    )
    
    model = get_peft_model(model, peft_config)
    
    # Check trainable parameters
    trainable, total = model.get_nb_trainable_parameters()
    print(f"LoRA Trainable Params: {trainable:,d} ({100 * trainable / total:.2f}% of total {total:,d})")

    # ─── 4. Dataset Processing ────────────────────────────────────────────────
    print(f"Loading dataset from: {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # Split into train and validation
    dataset_split = dataset.train_test_split(test_size=0.1)
    train_ds = dataset_split["train"].map(format_instruction_prompt)
    val_ds = dataset_split["test"].map(format_instruction_prompt)

    # ─── 5. Training Strategy (SFT) ───────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        fp16=True, # Mixed precision for faster training
        optim="paged_adamw_8bit",
        report_to="none" # Disable wandb for local script
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024, # Receipts aren't exceedingly long
        tokenizer=tokenizer,
        args=training_args,
    )

    # ─── 6. Execute Training ──────────────────────────────────────────────────
    print("🔥 Starting QLoRA Fine-tuning...")
    trainer.train()
    
    print(f"✅ Training complete. Saving LoRA adapter weights to {OUTPUT_DIR}/final_adapter")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")

if __name__ == "__main__":
    # Ensure dataset exists before running
    if os.path.exists(DATASET_PATH):
        main()
    else:
        print(f"Error: Dataset {DATASET_PATH} not found. Please create 'receipts_sft.jsonl' with 'ocr_text' and 'json_output' fields.")
