import argparse
from transformers import EncoderDecoderModel, BertTokenizer, Trainer, TrainingArguments
from data_processing import prepare_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./models/bert2bert_qgen")
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.model_name, args.model_name)

    tokenized_dataset = prepare_dataset(tokenizer, num_samples=args.num_samples)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("🚀 Iniciando el entrenamiento...")
    trainer.train()
    print("✅ Entrenamiento completado.")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
