## Imports
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
import time
import os
import shutil
from pathlib import Path
import logging
torch.cuda.empty_cache()

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_model(max_seq_length):
    # Configuration recommandée par Unsloth
    dtype = None  # Auto-détection du dtype
    
    logger.info("Loading model with Flash Attention 2...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=False,  # Pas de quantification 4-bit
        attn_implementation="flash_attention_2",  # Utilisation de Flash Attention 2
        rope_scaling={"type": "dynamic", "factor": 2.0},
        trust_remote_code=True
    )
    
    logger.info("Configuring LoRA with Unsloth optimizations...")
    model = FastLanguageModel.get_peft_model(
        model,
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
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=True,  # Utilisation de Rank-stabilized LoRA
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

    EOS_TOKEN = tokenizer.eos_token or '</s>'

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

def create_validation_dataset(dataset, val_size=0.1, seed=42):
    """Crée un dataset de validation à partir du dataset d'entraînement"""
    dataset = dataset.shuffle(seed=seed)
    val_size = int(len(dataset) * val_size)
    train_dataset = dataset.select(range(len(dataset) - val_size))
    val_dataset = dataset.select(range(len(dataset) - val_size, len(dataset)))
    return train_dataset, val_dataset

@torch.no_grad()
def evaluate_model(model, val_dataset, tokenizer, batch_size=1):
    """Évalue le modèle sur le dataset de validation"""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    try:
        for i in range(0, len(val_dataset), batch_size):
            batch = val_dataset[i:i + batch_size]
            inputs = tokenizer(batch['text'], truncation=True, padding=True, return_tensors='pt').to(model.device)
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs.loss, 'item') else outputs.loss['loss']
            total_loss += loss.item()
            total_batches += 1
            del outputs, loss, inputs
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Erreur pendant l'évaluation: {e}")
        model.train()
        return float('inf')
    
    model.train()
    return total_loss / total_batches if total_batches > 0 else float('inf')

def save_model(model, tokenizer, output_dir):
    """Sauvegarde le modèle avec Unsloth LoRA"""
    try:
        logger.info("Saving model with Unsloth...")
        # Pour Unsloth, on utilise directement save_pretrained
        model.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        
        # Sauvegarder le tokenizer
        tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully!")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def save_checkpoint(model, optimizer, scheduler, loss, step, checkpoint_dir, keep_last_n=3):
    """Sauvegarde un checkpoint du modèle"""
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{step}"
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Sauvegarder uniquement les adaptateurs LoRA
    model.save_pretrained(checkpoint_path)
    
    # Sauvegarder l'état de l'optimiseur et du scheduler
    torch.save({
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }, checkpoint_path / "training_state.pt")
    
    # Nettoyer les anciens checkpoints
    checkpoints = sorted([d for d in os.listdir(checkpoint_dir) 
                         if d.startswith("checkpoint_")])
    if len(checkpoints) > keep_last_n:
        for checkpoint in checkpoints[:-keep_last_n]:
            shutil.rmtree(os.path.join(checkpoint_dir, checkpoint))

class LoggingCallback:
    def __init__(self):
        self.best_loss = float('inf')
        self.start_time = time.time()
        self.start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        gpu_stats = torch.cuda.get_device_properties(0)
        self.max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        print(f"Starting training with {args.max_steps} steps")
        print(f"Initial GPU memory reserved: {self.start_gpu_memory} GB")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % args.logging_steps == 0:
            try:
                current_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
                memory_used = round(current_memory - self.start_gpu_memory, 3)
                elapsed_time = time.time() - self.start_time
                
                if state.log_history:
                    last_log = state.log_history[-1]
                    loss = last_log.get('loss', 0.0)
                    lr = last_log.get('learning_rate', 0.0)
                    grad_norm = last_log.get('grad_norm', 0.0)
                    
                    print(f"\nStep {state.global_step}/{args.max_steps} "
                          f"[{state.global_step/args.max_steps*100:.1f}%]")
                    print(f"Loss: {loss:.4f} | Grad Norm: {grad_norm:.4f} | LR: {lr:.2e}")
                    print(f"Memory: {current_memory}GB (+{memory_used}GB) | "
                          f"Time: {elapsed_time/60:.1f}min | "
                          f"Speed: {elapsed_time/state.global_step:.1f}s/it")
            except Exception as e:
                print(f"Step {state.global_step}: Logging failed - {str(e)}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            eval_loss = metrics['eval_loss']
            print(f"\nEvaluation - Step {state.global_step}")
            print(f"Eval Loss: {eval_loss:.4f}")
            
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                print(f"New best eval loss: {eval_loss:.4f}\n")

    def on_train_end(self, args, state, control, **kwargs):
        end_time = time.time()
        training_time = end_time - self.start_time
        final_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        memory_for_training = round(final_memory - self.start_gpu_memory, 3)
        
        print("\n=== Training Summary ===")
        print(f"Total time: {training_time/60:.1f} minutes")
        print(f"Final loss: {state.log_history[-1].get('loss', 0.0):.4f}")
        print(f"Best eval loss: {self.best_loss:.4f}")
        print("\n=== Memory Usage ===")
        print(f"Peak GPU memory: {final_memory}GB")
        print(f"Training memory: {memory_for_training}GB")
        print(f"Memory usage: {(final_memory/self.max_memory)*100:.1f}%")

def initialize_trainer(model, tokenizer, dataset, max_seq_length):
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=10,
        max_steps=1500,
        learning_rate=1e-4,
        fp16=False,  # Désactivé pour stabilité
        bf16=True,   # Utilisé si disponible
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        gradient_checkpointing=True,
        torch_compile=False,  # Désactivé pour éviter les problèmes
        max_grad_norm=1.0
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args
    )
    
    return trainer

def train_model(model, tokenizer, dataset, max_seq_length):
    # Créer le callback de logging
    logging_callback = LoggingCallback()
    
    trainer = initialize_trainer(model, tokenizer, dataset, max_seq_length)
    
    # Utiliser unsloth_train avec le callback
    trainer_stats = trainer.train(callbacks=[logging_callback])  # Ajouter le callback
    
    return trainer_stats

if __name__ == "__main__":
    start_time = time.time()
    max_seq_length = 2048
    model, tokenizer = initialize_model(max_seq_length)

    # Charger les données
    dataset = initialize_dataset(tokenizer, 'dataset2_comptable.csv')
    train_dataset, val_dataset = create_validation_dataset(dataset)

    # Créer le dossier pour les checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    
    # Statistiques GPU initiales
    gpu_stats = torch.cuda.get_device_properties(0)
    max_memory = gpu_stats.total_memory / 1024**3
    start_gpu_memory = torch.cuda.memory_reserved(0) / 1024**3
    
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    # Entraîner le modèle avec max_seq_length
    trainer_stats = train_model(model, tokenizer, train_dataset, max_seq_length)
    
    # Évaluer le modèle
    eval_loss = evaluate_model(model, val_dataset, tokenizer)
    print(f"Eval Loss = {eval_loss:.4f}")
    
    # Sauvegarder le modèle
    save_model(model, tokenizer, 'peft_model')

    # Statistiques finales
    end_time = time.time()
    used_memory = torch.cuda.max_memory_reserved() / 1024**3
    used_memory_for_lora = used_memory - start_gpu_memory
    used_percentage = used_memory / max_memory * 100
    lora_percentage = used_memory_for_lora / max_memory * 100
    print(f"{end_time - start_time} seconds used for training.")
    print(f"{round((end_time - start_time)/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Fusionner les poids LoRA avec le modèle de base et sauvegarder
    merged_model = model.merge_and_unload()
    save_model(merged_model, tokenizer, 'llama_model_merged')

    print("Modèle fusionné et sauvegardé avec succès.")
