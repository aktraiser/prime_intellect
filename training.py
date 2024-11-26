import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
import time

torch.cuda.empty_cache()

def initialize_model(max_seq_length):
    dtype = None  # Détection automatique. Float16 pour Tesla T4, V100, Bfloat16 pour Ampere+
    load_in_4bit = True  # Utiliser la quantification 4 bits pour réduire l'utilisation de la mémoire. Peut être False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token="hf_...",  # Utiliser un token si vous utilisez des modèles restreints comme meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choisissez un nombre > 0 ! Suggéré : 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Supporte n'importe quelle valeur, mais 0 est optimisé
        bias="none",     # Supporte n'importe quelle valeur, mais "none" est optimisé
        # [NOUVEAU] "unsloth" utilise 30% moins de VRAM, permet des batchs 2x plus grands !
        use_gradient_checkpointing="unsloth",  # True ou "unsloth" pour un contexte très long
        random_state=3407,
        use_rslora=False,  # Supporte Rank Stabilized LoRA
        loftq_config=None,  # Et LoftQ
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

    EOS_TOKEN = tokenizer.eos_token  # Doit ajouter EOS_TOKEN

    # Fonction pour formater les données
    def format_data(row):
        texte = row['Texte principal']
        question = row['Questions']
        reponse = row['Réponses']
        prompt = prompt_template.format(texte=texte, question=question) + reponse + EOS_TOKEN
        return {'text': prompt}

    # Create a list of formatted prompts
    formatted_data = []
    for _, row in df.iterrows():
        formatted_data.append({
            'text': prompt_template.format(
                texte=row['Texte principal'], 
                question=row['Questions']
            ) + row['Réponses'] + EOS_TOKEN
        })
    
    # Convert to dataset Hugging Face
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
        packing=False,  # Peut rendre l'entraînement 5x plus rapide pour les séquences courtes.
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # num_train_epochs=1,  # Définir pour un entraînement complet.
            max_steps=1000,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    return trainer

if __name__ == "__main__":
    start_time = time.time()
    max_seq_length = 2048
    model, tokenizer = initialize_model(max_seq_length)

    # Spécifiez le chemin vers votre fichier CSV
    csv_file = 'dataset_conseils_entreprises.csv'
    dataset = initialize_dataset(tokenizer, csv_file)

    trainer = initialize_trainer(model, tokenizer, dataset, max_seq_length)

    # Afficher les statistiques de mémoire GPU actuelles
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Mémoire maximale = {max_memory} GB.")
    print(f"{start_gpu_memory} GB de mémoire réservée.")

    trainer.train()

    end_time = time.time()
    # Afficher les statistiques finales de mémoire et de temps
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{end_time - start_time} secondes utilisées pour l'entraînement.")
    print(f"{round((end_time - start_time) / 60, 2)} minutes utilisées pour l'entraînement.")
    print(f"Pic de mémoire réservée = {used_memory} GB.")
    print(f"Pic de mémoire réservée pour l'entraînement = {used_memory_for_lora} GB.")
    print(f"Pourcentage de mémoire réservée par rapport à la mémoire maximale = {used_percentage} %.")
    print(f"Pourcentage de mémoire réservée pour l'entraînement par rapport à la mémoire maximale = {lora_percentage} %.")

    model.save_pretrained("llama_model")  # Sauvegarde locale
    tokenizer.save_pretrained("llama_model")
    # model.push_to_hub("votre_nom/lora_model", token="...")  # Sauvegarde en ligne
    # tokenizer.push_to_hub("votre_nom/lora_model", token="...")  # Sauvegarde en ligne