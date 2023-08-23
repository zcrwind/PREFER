from transformers.data.metrics import glue_compute_metrics

def text_classification_metrics(task_name, preds, labels):
    return {"acc": (preds == labels).mean()}

compute_metrics_mapping = {
    "cola": glue_compute_metrics,
    "mnli": text_classification_metrics,
    "mnli-mm": text_classification_metrics,
    "mrpc": glue_compute_metrics,
    "sst": text_classification_metrics,
    "imdb": text_classification_metrics,
    "agnews": text_classification_metrics,
    "sts-b": glue_compute_metrics,
    "qqp": glue_compute_metrics,
    "qnli": text_classification_metrics,
    "rte": text_classification_metrics,
    "wnli": glue_compute_metrics,
    "snli": text_classification_metrics,
    "mr": text_classification_metrics,
    "sst-5": text_classification_metrics,
    "subj": text_classification_metrics,
    "trec": text_classification_metrics,
    "cr": text_classification_metrics,
    "mpqa": text_classification_metrics,
}
