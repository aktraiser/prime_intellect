## Imports
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
import time
torch.cuda.empty_cache()

def initialize_model(max_seq_length):
    # Initialize tokenizer and model
    model_id = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        rope_scaling={"type": "dynamic", "factor": 2.0}
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)

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

def initialize_trainer(model, tokenizer, dataset, max_seq_length, training_args):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
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
    model, tokenizer, training_args, lr_scheduler = initialize_model(max_seq_length)

    # Charger les données à partir du fichier CSV
    dataset = initialize_dataset(tokenizer, 'dataset2_comptable.csv')

    trainer = initialize_trainer(model, tokenizer, dataset, max_seq_length, training_args)

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
