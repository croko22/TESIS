# src/evaluate.py

import evaluate # Necesitarás tener instalados 'datasets', 'evaluate', 'rouge_score', 'bert_score', 'nltk'
import numpy as np

class QuestionGenerationEvaluator:
    def __init__(self):
        self.bleu_metric = evaluate.load('bleu')
        self.rouge_metric = evaluate.load('rouge')
        self.meteor_metric = evaluate.load('meteor')
        self.bertscore_metric = evaluate.load('bertscore')

    def compute_metrics(self, predictions: list, references: list, lang="en"):
        """
        Calcula BLEU, ROUGE, METEOR y BERTScore para las preguntas generadas.

        Args:
            predictions (list): Una lista de strings con las preguntas generadas.
            references (list): Una lista de strings con las preguntas de referencia (ground truth).
            lang (str): Idioma para BERTScore (ej: "en" para inglés).

        Returns:
            dict: Un diccionario con los resultados de las métricas.
        """
        if not predictions or not references:
            return {"error": "Las listas de predicciones o referencias están vacías."}

        # BLEU
        bleu_results = self.bleu_metric.compute(predictions=predictions, references=references)
        
        # ROUGE
        rouge_results = self.rouge_metric.compute(predictions=predictions, references=references)
        
        # METEOR
        meteor_results = self.meteor_metric.compute(predictions=predictions, references=references)
        
        # BERTScore
        # BERTScore puede tomar tiempo para conjuntos de datos grandes
        bertscore_results = self.bertscore_metric.compute(predictions=predictions, references=references, lang=lang)
        
        # Calcula el promedio de precision, recall y f1 para BERTScore
        avg_bert_precision = np.mean(bertscore_results['precision'])
        avg_bert_recall = np.mean(bertscore_results['recall'])
        avg_bert_f1 = np.mean(bertscore_results['f1'])

        metrics = {
            "bleu": bleu_results["bleu"],
            "rouge1": rouge_results["rouge1"],
            "rouge2": rouge_results["rouge2"],
            "rougeL": rouge_results["rougeL"],
            "rougeLsum": rouge_results["rougeLsum"],
            "meteor": meteor_results["meteor"],
            "bertscore_precision": avg_bert_precision,
            "bertscore_recall": avg_bert_recall,
            "bertscore_f1": avg_bert_f1,
        }

        return metrics

# Ejemplo de uso (ejecutable si se guardara como evaluate.py y se llamara desde otro script)
if __name__ == "__main__":
    # Datos de ejemplo
    generated_questions_example = [
        "What is the capital of France?",
        "Who created the Transformer model?",
        "What is the color of the sky?"
    ]
    target_questions_example = [
        "What is the capital of France?",
        "Who developed the Transformer architecture?",
        "What color is the sky today?"
    ]

    evaluator = QuestionGenerationEvaluator()
    results = evaluator.compute_metrics(generated_questions_example, target_questions_example)
    
    print("Resultados de las Métricas:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")