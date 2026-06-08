from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf


def create_model(num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    model = TFAutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=num_labels
    )

    return tokenizer, model