from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class QuestionGeneratorModel:
    def __init__(self, model_name="t5-base"): 
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model