from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_trained_model(model_path: str):
    print(f"Cargando modelo desde: {model_path}")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    return model, tokenizer