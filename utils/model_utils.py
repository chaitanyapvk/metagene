import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_model(model_config, return_model=False):
    """
    Load the model based on the input configuration

    Parameters
    ----------
    model_config : dict
        model configuration. keys: model_path, num_classes

    return_model : bool, optional
        return model, tokenizer, device, by default False

    Returns
    -------
    model, tokenizer, device: optional

    """

    global model, tokenizer



    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_path"], do_lower_case=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["model_path"], num_labels=model_config["num_classes"]
    )

    print(f'{ model_config["model_path"]} loaded')

    

    if return_model:
        return model, tokenizer, 


def compute_metrics(eval_preds):
    """
    A function for computing the metrics for the model at the end of each epoch

    Parameters
    ----------
    eval_preds : tuple
        a tuple that includes model predictions and correct labels

    Returns
    -------
    :dict
        dictionary of metrics
    """

    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)

    metrics = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    return metrics