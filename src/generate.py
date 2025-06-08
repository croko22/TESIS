import argparse
import torch
from model import load_trained_model

def generate_question(context: str, intent: str, model, tokenizer):
    """
    Genera una pregunta usando el modelo fine-tuned.
    """
    input_text = f"generar pregunta para {intent}: {context}"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    outputs = model.generate(
        inputs.input_ids,
        max_length=128,
        num_beams=5,  # Usar beam search para generar mejores preguntas
        early_stopping=True
    )
    
    generated_question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_question

def main():
    parser = argparse.ArgumentParser(description="Generar una pregunta con un modelo T5 entrenado.")
    parser.add_argument("--model_path", type=str, required=True, help="Ruta al directorio del modelo entrenado.")
    parser.add_argument("--context", type=str, required=True, help="El texto de contexto para generar la pregunta.")
    parser.add_argument("--intent", type=str, required=True, help="El intento deseado (e.g., 'what', 'why', 'how').")
    
    args = parser.parse_args()

    # Cargar modelo
    model, tokenizer = load_trained_model(args.model_path)

    # Generar pregunta
    question = generate_question(args.context, args.intent, model, tokenizer)
    
    print("\n" + "="*20)
    print(f"Contexto: {args.context}")
    print(f"Intento: {args.intent}")
    print("-" * 20)
    print(f"Pregunta Generada: {question}")
    print("="*20)