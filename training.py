import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import PeftModel
import time

torch.cuda.empty_cache()

def initialize_model(max_seq_length, load_in_4bit=False):
    dtype = torch.float16  # Utiliser float16 pour la pleine précision

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

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choisissez un nombre > 0 ! Suggéré : 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # 0 est optimisé
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

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
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=1000,  # Augmenter le nombre d'étapes d'entraînement
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            save_strategy="steps",
            save_steps=100,  # Sauvegarder périodiquement
            save_total_limit=2,  # Garder les deux derniers points de contrôle
        ),
    )

    return trainer

if __name__ == "__main__":
    start_time = time.time()
    max_seq_length = 2048
    load_in_4bit = False  # Désactiver la quantification en 4 bits

    model, tokenizer = initialize_model(max_seq_length, load_in_4bit)

    # Charger les données
    csv_file = 'dataset_conseils_entreprises.csv'
    dataset = initialize_dataset(tokenizer, csv_file)

    # Initialiser le formateur
    trainer = initialize_trainer(model, tokenizer, dataset, max_seq_length)

    # Statistiques mémoire GPU
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Mémoire maximale = {max_memory} GB.")
    print(f"{start_gpu_memory} GB de mémoire réservée.")

    # Entraînement
    trainer.train()

    # Sauvegarde du modèle LoRA
    trainer.save_model()

    # Fusion des poids LoRA avec le modèle de base
    print("Fusion des poids LoRA avec le modèle de base...")
    peft_model = PeftModel.from_pretrained(model, "outputs")
    merged_model = peft_model.merge_and_unload()

    # Préparer le modèle pour l'inférence
    FastLanguageModel.for_inference(merged_model)

    # Sauvegarde du modèle fusionné et du tokenizer
    merged_model.save_pretrained("llama_model_merged", safe_serialization=True)
    tokenizer.save_pretrained("llama_model_merged")
    print("Modèle fusionné et sauvegardé.")

    # Validation locale
    print("Validation locale du modèle sauvegardé...")
    from unsloth import FastLanguageModel

    # Charger le modèle fusionné en utilisant unsloth
    merged_model, merged_tokenizer = FastLanguageModel.from_pretrained(
        model_name="llama_model_merged",
        max_seq_length=max_seq_length,
        load_in_4bit=False,
    )

    # Préparer le modèle pour l'inférence
    FastLanguageModel.for_inference(merged_model)

    # Test simple
    inputs = merged_tokenizer("Ceci est un test.", return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    outputs = merged_model.generate(**inputs, max_new_tokens=50)
    print("Exemple de génération :", merged_tokenizer.decode(outputs[0]))

    end_time = time.time()
    print(f"{end_time - start_time} secondes utilisées pour l'entraînement et la fusion.")
