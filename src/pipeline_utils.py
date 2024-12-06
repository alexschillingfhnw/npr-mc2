from datasets import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    get_scheduler,
)
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


def get_device():
    """
    Determine the best available device (CUDA, MPS, or CPU).

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print(f"Selected device: {device}")
    return device


# ====================
# Tokenization
# ====================
def tokenize(batch, tokenizer):
    return tokenizer(batch['sentence'], padding="max_length", truncation=True, max_length=128)


def convert_and_tokenize(df, tokenizer):
    ds = Dataset.from_pandas(df)
    ds = ds.map(lambda batch: tokenize(batch, tokenizer), batched=True)
    required_columns = ["input_ids", "attention_mask", "label"]
    ds = ds.remove_columns([col for col in ds.column_names if col not in required_columns])
    return ds


# ====================
# Compute Metrics
# ====================
def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), dim=1).numpy()
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# ====================
# Custom Callbacks
# ====================
class LossTrackerCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Track training and evaluation losses
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])

        # Plot dynamically with smoothing
        if len(self.train_losses) > 2:  # At least three points for smoothing
            clear_output(wait=True)
            train_smoothed = moving_average(self.train_losses)
            eval_smoothed = moving_average(self.eval_losses) if len(self.eval_losses) > 2 else self.eval_losses
            plt.figure(figsize=(10, 6))
            plt.plot(train_smoothed, label="Training Loss (Smoothed)", marker='o')
            plt.plot(eval_smoothed, label="Validation Loss (Smoothed)", marker='o')
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss (Smoothed)")
            plt.legend()
            plt.grid(True)
            plt.show()


class EarlyStoppingByValLoss(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience=2, early_stopping_threshold=0.001):
        super().__init__()
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if "eval_loss" in metrics:
            current_loss = metrics["eval_loss"]
            if (
                state.best_metric is None
                or current_loss < state.best_metric - self.early_stopping_threshold
            ):
                state.best_metric = current_loss
                state.best_model_checkpoint = f"{args.output_dir}/checkpoint-{state.global_step}"
                state.early_stopping_patience = 0
            else:
                state.early_stopping_patience += 1
                if state.early_stopping_patience >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {self.early_stopping_patience} evaluations.")
                    control.should_training_stop = True


class TimeRemainingCallback(TrainerCallback):
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = None

    def format_time(self, seconds):
        """Converts seconds to a formatted string of hours and minutes."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}min"

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step > 0 and state.global_step % 100 == 0:  # Print every 100 steps
            elapsed_time = time.time() - self.start_time
            steps_completed = state.global_step
            steps_remaining = self.total_steps - steps_completed
            time_per_step = elapsed_time / steps_completed
            estimated_time_remaining = steps_remaining * time_per_step

            print(
                f"Progress: {steps_completed}/{self.total_steps} | "
                f"Elapsed Time: {self.format_time(elapsed_time)} | "
                f"Estimated Time Remaining: {self.format_time(estimated_time_remaining)}"
            )


# ====================
# Moving Average for Smoothing
# ====================
def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# ====================
# Train Model
# ====================
def train_model(train_ds, validation_ds, total_steps, device):
    """
    Train a BERT model and return training/evaluation metrics in memory.
    """
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

    training_args = TrainingArguments(
        output_dir="./tmp/",  # Temporary directory
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=50,  # Save every 50 steps
        logging_steps=25,  # Log every 25 steps for better insights
        report_to=["none"],  # Disable external log reporting
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        save_total_limit=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        compute_metrics=compute_metrics,
        callbacks=[
            LossTrackerCallback(),
            EarlyStoppingByValLoss(early_stopping_patience=2, early_stopping_threshold=0.001),
            TimeRemainingCallback(total_steps=total_steps),
        ],
    )

    print("Starting training...")
    trainer.train()
    print("Calculating final training loss...")
    train_loss = trainer.evaluate(train_ds)["eval_loss"]
    print("Training complete. Evaluating...")
    eval_results = trainer.evaluate()

    metrics = {
        "train_loss": train_loss,
        "eval_loss": eval_results["eval_loss"],
        "eval_accuracy": eval_results["eval_accuracy"],
        "eval_precision": eval_results["eval_precision"],
        "eval_recall": eval_results["eval_recall"],
        "eval_f1": eval_results["eval_f1"],
    }

    return trainer, metrics


# ====================
# Evaluate Model
# ====================
def evaluate_model(trainer, test_ds):
    results = trainer.evaluate(test_ds)
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    return results


# ====================
# Plots
# ====================
def plot_metrics(metrics_df, x_col="split_size", y_cols=None, title="Model Quality by Training Data Size"):
    """
    Plot specified metrics against training data split size.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing metrics for each training split size.
        x_col (str): Column name for x-axis (e.g., 'split_size').
        y_cols (list): List of column names to plot on the y-axis.
                       Defaults to ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1'].
        title (str): Title for the plot.

    Returns:
        None: Displays the plot.
    """
    if y_cols is None:
        y_cols = ["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"]
    
    plt.figure(figsize=(10, 6))
    for y_col in y_cols:
        plt.plot(metrics_df[x_col], metrics_df[y_col], marker="o", label=y_col.replace("_", " ").title())
    
    plt.title(title)
    plt.xlabel("Training Data Split Size")
    plt.ylabel("Metric Value")
    plt.ylim(0.7, 1)  # Metrics are typically scaled between 0 and 1
    plt.legend()
    plt.grid(True)
    plt.show()