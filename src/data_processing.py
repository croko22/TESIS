from datasets import load_dataset 
from transformers import T5Tokenizer

def get_intent(question: str) -> str:
    """
    Clasifica una pregunta para obtener un 'intento' basado en su primera palabra.
    Esta es una heurística simple. Puedes mejorarla como parte de tu tesis.
    """
    question = question.lower().strip()
    first_word = question.split()[0]
    
    intent_words = ["what", "who", "when", "where", "why", "how", "which"]
    return first_word if first_word in intent_words else "other"

def prepare_dataset(model_name: str, dataset_name: str = "squad", num_samples: int = None):
    """
    Carga, procesa y tokeniza el dataset para el entrenamiento de T5.
    """
    print(f"Cargando tokenizer para el modelo: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    print(f"Cargando dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name, split="train")

    if num_samples:
        print(f"Seleccionando un subconjunto de {num_samples} ejemplos.")
        dataset = dataset.select(range(num_samples))

    def tokenize_function(examples):
        contexts = examples["context"]
        questions = examples["question"]
        
        inputs = []
        for context, question in zip(contexts, questions):
            intent = get_intent(question)
            # El formato del input es crucial para que el modelo aprenda la tarea
            formatted_input = f"generar pregunta para {intent}: {context}"
            inputs.append(formatted_input)
            
        # El target es la pregunta que el modelo debe aprender a generar
        targets = questions

        # Tokenizar inputs y targets
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length")
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Preprocesando y tokenizando el dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    return tokenized_dataset, tokenizer