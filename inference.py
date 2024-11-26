import torch
from unsloth import FastLanguageModel
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()

def initialize_model(max_seq_length):
    dtype = None  # Détection automatique. Float16 pour Tesla T4, V100, Bfloat16 pour Ampere+
    load_in_4bit = True  # Utiliser la quantification 4 bits pour réduire l'utilisation de la mémoire. Peut être False.

    base_model_name = "unsloth/Meta-Llama-3.1-8B"  # Même modèle de base que celui utilisé lors de l'entraînement

    # Charger le modèle de base et appliquer les poids LoRA entraînés
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        peft_model_id="llama_model",  # Chemin vers votre modèle entraîné
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token="hf_...",  # Utiliser un token si vous utilisez des modèles restreints
    )

    model.eval()  # Mettre le modèle en mode évaluation
    return model, tokenizer

def interactive_evaluation(model, tokenizer):
    prompt_template = """Tu es un expert en fiscalité, comptabilité, ton objectif et d'accompagner les entreprises dans leur enjeux. 

### Texte principal:
{texte}

### Question:
{question}

### Réponse:
"""
    while True:
        texte = input("Entrez le texte principal (ou 'exit' pour quitter) : ")
        if texte.lower() == 'exit':
            break
        question = input("Entrez la question : ")
        prompt = prompt_template.format(texte=texte, question=question)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
        output_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        print("Réponse:", output_text)
        print("-" * 80)