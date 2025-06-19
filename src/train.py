from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader
import torch

def train_model(model, tokenizer, train_dataset, eval_dataset, num_epochs=3, learning_rate=5e-5, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        model.eval()
        total_eval_loss = 0
        for batch in eval_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            total_eval_loss += outputs.loss.item()
        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        print(f"Epoch {epoch+1} - Validation Loss: {avg_eval_loss:.4f}")

    model.save_pretrained("./fine_tuned_qg_model")
    tokenizer.save_pretrained("./fine_tuned_qg_model")


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from functools import partial
    from src.data_processing import highlight_answer, prepare_instruction_dataset

    squad_dataset = load_dataset('squad')
    
    def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
        contexts = examples['context']
        questions = examples['question']
        highlighted_contexts = [
            highlight_answer({'context': c, 'answers': {'text': [examples['answers']['text'][i]]}})['answer_highlighted_context']
            for i, c in enumerate(contexts)
        ]
        instruction_prompts = [
            prepare_instruction_dataset({'answer_highlighted_context': hc})['instruction_prompt']
            for hc in highlighted_contexts
        ]
        model_inputs = tokenizer(instruction_prompts, max_length=max_input_length, truncation=True)
        labels = tokenizer(questions, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    max_input_length = 512
    max_target_length = 128

    model_name_for_training = "t5-base"
    model_instance_for_training = QuestionGeneratorModel(model_name=model_name_for_training)
    tokenizer_for_training = model_instance_for_training.get_tokenizer()
    model_for_training = model_instance_for_training.get_model()

    train_data = squad_dataset['train'].map(
        partial(preprocess_function, 
                tokenizer=tokenizer_for_training, 
                max_input_length=max_input_length, 
                max_target_length=max_target_length),
        batched=True,
        remove_columns=squad_dataset["train"].column_names
    )
    train_dataset = train_data.with_format("torch")
    eval_data = squad_dataset['validation'].map(
        partial(preprocess_function, 
                tokenizer=tokenizer_for_training, 
                max_input_length=max_input_length, 
                max_target_length=max_target_length),
        batched=True,
        remove_columns=squad_dataset["validation"].column_names
    )
    eval_dataset = eval_data.with_format("torch")

    train_model(model_for_training, tokenizer_for_training, train_dataset, eval_dataset, num_epochs=3, learning_rate=5e-5, batch_size=8)