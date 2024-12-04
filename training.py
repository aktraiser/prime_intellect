## Imports
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import time

torch.cuda.empty_cache()

def initialize_model(max_seq_length):
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage

    # Charger le modèle de base avec la configuration mise à jour
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # Remplacer use_flash_attention par attn_implementation
        attn_implementation="flash_attention_2",
        rope_scaling={"type": "dynamic", "factor": 2.0},
        trust_remote_code=True
    )

    # Configuration LoRA optimisée pour Unsloth
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Rang de la matrice LoRA
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_alpha=32,    # Scaling factor
        lora_dropout=0.0, # Unsloth recommande 0.0 pour de meilleures performances
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=True,  # Rank-stabilized LoRA
        # Suppression de target_r qui n'est pas supporté
        # Ajout des configurations LoFTQ
        loftq_config={
            "loftq_bits": 4,
            "loftq_iter": 1
        }
    )

    return model, tokenizer

def initialize_dataset(tokenizer, csv_file, max_seq_length):
    # Charger le fichier CSV avec le bon séparateur et gestion des quotes
    try:
        df = pd.read_csv(
            csv_file, 
            sep=';',
            quoting=pd.io.common.QUOTE_MINIMAL,  # Gestion des champs contenant des séparateurs
            encoding='utf-8'
        )
        
        # Vérifier les colonnes requises
        expected_columns = ['main_text', 'questions', 'answers', 'title']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print("Colonnes présentes dans le CSV:", df.columns.tolist())
            raise ValueError(f"Colonnes requises manquantes : {missing_columns}")
            
        # Nettoyer les données
        for col in df.columns:
            if df[col].dtype == 'object':
                # Supprimer les sauts de ligne et les espaces en trop
                df[col] = df[col].str.strip().replace(r'\s+', ' ', regex=True)
                
        return df
        
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV: {str(e)}")
        print("Tentative avec un autre encodage...")
        
        # Essayer avec un autre encodage si utf-8 échoue
        df = pd.read_csv(
            csv_file,
            sep=';',
            quoting=pd.io.common.QUOTE_MINIMAL,
            encoding='latin-1'
        )
        return df

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
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,  # Augmenté pour compenser batch_size=1
            warmup_steps=10,
            max_steps=1500,
            learning_rate=1e-4,  # Réduit pour plus de stabilité
            fp16=False,  # Désactivé pour éviter les problèmes numériques
            bf16=True,  # Utilisé si disponible
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",  # Changé pour une meilleure convergence
            seed=3407,
            output_dir="outputs",
            gradient_checkpointing=True,
            torch_compile=False,  # Désactivé pour éviter les problèmes de compilation
            max_grad_norm=1.0,  # Ajouté pour la stabilité
        ),
    )
    return trainer

def save_model(model, tokenizer, output_dir):
    # Supprimer les attributs spécifiques à l'entraînement
    if hasattr(model, 'module'):
        model = model.module
    
    # Sauvegarder avec la configuration complète
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="10GB"
    )
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    start_time = time.time()
    max_seq_length = 2048
    model, tokenizer = initialize_model(max_seq_length)

    # Charger les données à partir du fichier CSV
    dataset = initialize_dataset(tokenizer, 'dataset2_comptable.csv', max_seq_length)

    trainer = initialize_trainer(model, tokenizer, dataset, max_seq_length)

    # Afficher les statistiques de mémoire GPU
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Entraîner le modèle
    trainer.train()
    
    end_time = time.time()
    # Afficher les statistiques finales de mémoire et de temps
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{end_time - start_time} seconds used for training.")
    print(f"{round((end_time - start_time)/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Sauvegarder l'adaptateur LoRA
    save_model(model, tokenizer, 'peft_model')

    # Fusionner les poids LoRA avec le modèle de base et sauvegarder
    merged_model = model.merge_and_unload()
    save_model(merged_model, tokenizer, 'llama_model_merged')

    print("Modèle fusionné et sauvegardé avec succès.")
