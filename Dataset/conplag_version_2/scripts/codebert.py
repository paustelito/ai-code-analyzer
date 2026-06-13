from transformers import AutoTokenizer, AutoModelForSequenceClassification


def create_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=2,
        output_hidden_states=True
    )

    return tokenizer, model