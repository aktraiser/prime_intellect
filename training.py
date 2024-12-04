## Imports
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, EvalPrediction
from unsloth import is_bfloat16_supported
import time
from evaluate import load
import numpy as np
torch.cuda.empty_cache()

# At global scope
global_tokenizer = None

def initialize_model(max_seq_length):
 dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
 load_in_4bit = True  # Use 4bit quantization to reduce memory usage

 # Charger le modèle de base avec la configuration mise à jour
 model, tokenizer = FastLanguageModel.from_pretrained(
     model_name="unsloth/Meta-Llama-3.1-8B",
     max_seq_length=max_seq_length,
     dtype=dtype,
     load_in_4bit=load_in_4bit,
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
     lora_alpha=32,    # Facteur de scaling
     lora_dropout=0.0, # Unsloth recommande 0.0 pour de meilleures performances
     bias="none",
     use_gradient_checkpointing=True,
     random_state=3407,
     use_rslora=True,  # Rank-stabilized LoRA
     loftq_config={
         "loftq_bits": 4,
         "loftq_iter": 1
     }
 )

 return model, tokenizer

def initialize_dataset(tokenizer, csv_file):
    # Charger le fichier CSV
    df = pd.read_csv(csv_file)

        # Définir le format du prompt
    prompt_template = """Tu es un expert comptable spécialisé dans le conseil aux entreprises. En te basant uniquement sur le contexte fourni, réponds à la question de manière précise et professionnelle.

### Contexte:
{title}

### Document de référence:
{texte}

### Question du client:
{question}

### Instructions:
- Base ta réponse uniquement sur les informations fournies dans le document
- Prend en compte le title de la reponse pour avoir le contexte
- Si plusieurs contexte sont identique structure une réponse prennant en compte l'ensemble de leurs données de response
- Fournis une réponse claire et structurée
- Utilise un langage professionnel adapté au contexte comptable
- Si une information n'est pas disponible dans le contexte, indique-le clairement
- Commence ta réponse par un bref résumé de la situation
- Structure ta réponse avec des points clés si nécessaire
- Cite les toujours les références spécifiques du document
- Termine par une conclusion ou recommandation si approprié
- En cas de concepts techniques, fournis une brève explication
- Explique les termes techniques
- Si plusieurs options sont possibles, présente-les de manière structurée

### Réponse de l'expert:
"""

    EOS_TOKEN = tokenizer.eos_token or '<|endoftext|>'

        # Formater les données
    formatted_data = []
    for _, row in df.iterrows():
        formatted_data.append({
            'text': prompt_template.format(
                title=row['title'],
                texte=row['main_text'],
                question=row['questions']
            ) + row['answers'] + EOS_TOKEN
        })

        # Créer le dataset Hugging Face
    dataset = Dataset.from_dict({'text': [d['text'] for d in formatted_data]})

    return dataset

def initialize_trainer(model, tokenizer, dataset, max_seq_length):
    global global_tokenizer
    global_tokenizer = tokenizer
    # Séparer le dataset en train et validation
    dataset_dict = dataset.train_test_split(test_size=0.1, seed=3407)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],  # Ajout du dataset de validation
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        compute_metrics=compute_metrics,  # Ajout des métriques
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
            evaluation_strategy="steps",    # Évaluer périodiquement
            eval_steps=100,                # Évaluer tous les 100 steps
            save_strategy="steps",         # Sauvegarder périodiquement
            save_steps=100,                # Sauvegarder tous les 100 steps
            load_best_model_at_end=True,   # Charger le meilleur modèle à la fin
            metric_for_best_model="bertscore_f1",  # Métrique pour sélectionner le meilleur modèle
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

def compute_metrics(eval_pred: EvalPrediction):
    # Use global tokenizer
    predictions = global_tokenizer.batch_decode(eval_pred.predictions, skip_special_tokens=True)
    references = global_tokenizer.batch_decode(eval_pred.label_ids, skip_special_tokens=True)
    
    # Charger les métriques
    rouge = load("rouge")
    bertscore = load("bertscore")
    
    # Calculer ROUGE
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    
    # Calculer BERTScore
    bertscore_results = bertscore.compute(
        predictions=predictions, 
        references=references, 
        lang="fr",
        batch_size=8
    )
    
    metrics = {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bertscore_f1": np.mean(bertscore_results["f1"])
    }
    
    return metrics

if __name__ == "__main__":
    start_time = time.time()
    max_seq_length = 2048
    model, tokenizer = initialize_model(max_seq_length)

    # Charger les données à partir du fichier CSV
    dataset = initialize_dataset(tokenizer, 'dataset2_comptable.csv')

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
