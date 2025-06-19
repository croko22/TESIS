

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class QuestionGenerator:
    def __init__(self, model_path="./fine_tuned_qg_model"): 
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_question(self, context_with_highlight):
        inputs = self.tokenizer(
            context_with_highlight,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )

        outputs = self.model.generate(
            inputs['input_ids'].to(self.device),
            max_length=128,             
            num_beams=8,                
            no_repeat_ngram_size=3,     
            length_penalty=0.7,         
            early_stopping=True         
        )

        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return question


if __name__ == "__main__":
    
    q_gen = QuestionGenerator(model_path="./fine_tuned_qg_model") 
    
    example_context = "The capital of France is <h>Paris</h>."
    generated_question = q_gen.generate_question(example_context)
    print(f"Context: {example_context}")
    print(f"Generated Question: {generated_question}")