import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer
import time

torch.cuda.empty_cache()

def initialize_model(max_seq_length, load_in_4bit=False):
    dtype = torch.float16  # Utiliser float16 pour économiser de la mémoire

    # Charger le modèle de base
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Ajouter un token de padding si nécessaire
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Définir la configuration LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Préparer le modèle pour l'entraînement avec LoRA
    model = get_peft_model(model, peft_config)

    return model, tokenizer

def initialize_dataset(tokenizer, csv_file):
    # Charger le fichier CSV
    df = pd.read_csv(csv_file)

    # Définir le format du prompt
    prompt_template = """Vous êtes un expert en fiscalité. Répondez à la question suivante en vous basant sur le texte fourni.

### Texte principal:
{texte}

### Question:
{question}

### Réponse:
"""

    EOS_TOKEN = tokenizer.eos_token

    # Formater les données
    formatted_data = []
    for _, row in df.iterrows():
        formatted_data.append({
            'text': prompt_template.format(
                texte=row['Texte principal'],
                question=row['Questions']
            ) + row['Réponses'] + EOS_TOKEN
        })

    # Convertir en dataset Hugging Face
    dataset = Dataset.from_dict({'text': [d['text'] for d in formatted_data]})

    return dataset

def initialize_trainer(model, tokenizer, dataset, max_seq_length):
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=1000,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        dataloader_num_workers=2
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    return trainer

if __name__ == "__main__":
    start_time = time.time()
    max_seq_length = 2048
    load_in_4bit = False

    # Étape 1: Initialiser le modèle et le tokenizer
    model, tokenizer = initialize_model(max_seq_length, load_in_4bit)

    # Étape 2: Charger les données
    csv_file = 'dataset_conseils_entreprises.csv'
    dataset = initialize_dataset(tokenizer, csv_file)

    # Étape 3: Initialiser le formateur
    trainer = initialize_trainer(model, tokenizer, dataset, max_seq_length)

    # Étape 4: Entraîner le modèle
    trainer.train()

    # Étape 5: Sauvegarder l'adaptateur LoRA
    peft_model_id = "lora_adapter"
    model.save_pretrained(peft_model_id)

    # Étape 6: Fusionner les poids LoRA avec le modèle de base
    print("Fusion des poids LoRA avec le modèle de base...")

    # Recharger le modèle de base
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B",
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=False,
    )

    # Charger l'adaptateur LoRA dans le modèle de base
    model_to_merge = PeftModel.from_pretrained(base_model, peft_model_id)

    # Fusionner et décharger les poids LoRA
    model_to_merge = model_to_merge.merge_and_unload()

    # Préparer le modèle pour l'inférence
    FastLanguageModel.for_inference(model_to_merge)

    # Sauvegarder le modèle fusionné et le tokenizer
    model_to_merge.save_pretrained("llama_model_merged", safe_serialization=True)
    tokenizer.save_pretrained("llama_model_merged")
    print("Modèle fusionné et sauvegardé.")

    # Étape 7: Validation locale du modèle fusionné
    print("Validation locale du modèle sauvegardé...")

    # Charger le modèle fusionné pour l'inférence
    merged_model, merged_tokenizer = FastLanguageModel.from_pretrained(
        model_name="llama_model_merged",
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=False,
    )

    # Préparer le modèle pour l'inférence
    FastLanguageModel.for_inference(merged_model)

    # Test de génération simple
    inputs = merged_tokenizer("Ceci est un test.", return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    outputs = merged_model.generate(**inputs, max_new_tokens=50)
    print("Exemple de génération :", merged_tokenizer.decode(outputs[0]))

    end_time = time.time()
    print(f"{end_time - start_time} secondes utilisées pour l'entraînement et la fusion.")
