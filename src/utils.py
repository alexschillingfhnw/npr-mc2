from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

import torch

import os
import time
import numpy as np
import pandas as pd
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.manifold import TSNE


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
# Create Nested Data Splits
# ====================
def create_nested_splits(data, split_sizes, num_sets=10, base_folder="../data/nested_splits"):
    """
    Create multiple sets of nested splits with shuffling and save each set in a separate folder.
    Args:
        data (pd.DataFrame): Input dataframe.
        split_sizes (list): List of split ratios (e.g., [0.01, 0.1]).
        num_sets (int): Number of shuffled split sets to create.
        base_folder (str): Base directory to store nested splits.
    Returns:
        dict: Dictionary with sets of nested splits.
    """
    os.makedirs(base_folder, exist_ok=True)  # Ensure the base folder exists
    all_nested_splits = {}

    for set_idx in range(num_sets):
        set_folder = os.path.join(base_folder, f"set_{set_idx + 1}")
        os.makedirs(set_folder, exist_ok=True)  # Create folder for each set
        
        shuffled_data = data.sample(frac=1, random_state=None).reset_index(drop=True)  # Shuffle data
        nested_splits = {}

        for size in split_sizes:
            sample_size = int(len(shuffled_data) * size)
            nested_splits[f"{int(size * 100)}%"] = shuffled_data.iloc[:sample_size]
            
            # Save each split to the respective set folder
            split_filename = os.path.join(set_folder, f"split_{int(size * 100)}.csv")
            nested_splits[f"{int(size * 100)}%"].to_csv(split_filename, index=False)
        
        all_nested_splits[f"Set_{set_idx + 1}"] = nested_splits
        print(f"Created Set {set_idx + 1} with nested splits.")
    
    return all_nested_splits


# --------------------
# Tokenization
# --------------------
def tokenize_dataset(df, tokenizer, max_length=128):
    """
    Tokenizes the input DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'sentence' and 'label' columns.
        tokenizer (BertTokenizer): Pretrained BERT tokenizer.
        max_length (int): Maximum sequence length.

    Returns:
        Dataset: Tokenized HuggingFace Dataset.
    """
    dataset = Dataset.from_pandas(df[['sentence', 'label']])
    tokenized = dataset.map(
        lambda x: tokenizer(
            x['sentence'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        ),
        batched=True
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized

# --------------------
# Metrics Computation
# --------------------
def compute_metrics(pred):
    """
    Computes evaluation metrics.

    Args:
        pred: Predictions from the model.

    Returns:
        dict: Accuracy, precision, recall, and F1-score.
    """
    preds = torch.argmax(torch.tensor(pred.predictions), dim=1).numpy()
    labels = pred.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# --------------------
# Model Training
# --------------------
def train_and_evaluate(train_dataset, eval_dataset, output_dir, epochs=3, batch_size=8, learning_rate=2e-5, weight_decay=0.01):
    """
    Trains the BERT model and evaluates it.

    Args:
        train_dataset (Dataset): Tokenized training dataset.
        eval_dataset (Dataset): Tokenized evaluation dataset.
        output_dir (str): Directory to save model checkpoints.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and evaluation.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        dict: Evaluation metrics.
    """
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=1,  # Keep only the best model
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"Training model in {output_dir}...")
    trainer.train()

    print("Evaluating model...")
    eval_metrics = trainer.evaluate()

    return eval_metrics


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
# Generate Embeddings
# ====================
def generate_embeddings(sentences, model_name="all-MiniLM-L6-v2", device="cpu"):
    """
    Generate embeddings for a list of sentences using a sentence-transformer model.

    Args:
        sentences (list): List of sentences to embed.
        model_name (str): Pretrained sentence-transformer model.

    Returns:
        np.ndarray: Sentence embeddings.
    """
    model = SentenceTransformer(model_name).to(device)
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings


def calculate_similarity(embeddings, reference_index=0):
    """
    Calculate cosine similarity of embeddings with respect to a reference embedding.

    Args:
        embeddings (np.ndarray): Sentence embeddings.
        reference_index (int): Index of the reference embedding for similarity calculation.

    Returns:
        np.ndarray: Cosine similarity scores.
    """
    reference_embedding = embeddings[reference_index].reshape(1, -1)
    similarities = cosine_similarity(reference_embedding, embeddings)
    return similarities.flatten()


def get_and_print_most_similar_sentences(embeddings, sentences, reference_index, top_k=5):
    """
    Print the top-k most similar sentences to a reference sentence and also return their embeddings for visualization.
    """
    similarities = calculate_similarity(embeddings, reference_index=reference_index)

    # Display top-5 most similar sentences
    sorted_indices = np.argsort(-similarities)  # Sort in descending order
    print("Top-5 most similar sentences:")
    for idx in sorted_indices[:5]:
        print(f"Similarity: {similarities[idx]:.4f}, Sentence: {sentences[idx]}")

    # Get embeddings of the top-k most similar sentences
    top_k_embeddings = embeddings[sorted_indices[:top_k]]
    return top_k_embeddings


def get_min_max_distance(embeddings, sentences):
    # Calculate pairwise cosine distances
    distances = cosine_distances(embeddings)

    # Find the sentence with the furthest distance to any other sentence
    max_distance_idx = np.unravel_index(np.argmax(distances, axis=None), distances.shape)
    max_distance_sentence_idx = max_distance_idx[0]

    # Find the sentence closest to the most others (minimum average distance)
    average_distances = distances.mean(axis=1)
    min_average_distance_idx = np.argmin(average_distances)

    # Results
    max_distance_sentence = sentences[max_distance_sentence_idx]
    min_average_distance_sentence = sentences[min_average_distance_idx]

    print(f"Sentence with the furthest distance to any other sentence:\n'{max_distance_sentence}'")
    print()
    print(f"Sentence closest to the most others (minimum average distance):\n'{min_average_distance_sentence}'")


# ====================
# Plots
# ====================
def plot_similarity_density(embeddings, labeled_labels):
    """
    Plot density of cosine similarities within classes.

    Args:
        embeddings (np.array): Embeddings of the labeled dataset.
        labeled_labels (list): Labels of the labeled dataset.
    """

    # Generate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Extract same-class similarities
    blue_indices = [i for i, label in enumerate(labeled_labels) if label == 1]
    red_indices = [i for i, label in enumerate(labeled_labels) if label == 0]

    blue_to_blue = similarity_matrix[np.ix_(blue_indices, blue_indices)].flatten()
    red_to_red = similarity_matrix[np.ix_(red_indices, red_indices)].flatten()

    # Density plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(blue_to_blue, label="Blue-to-Blue Similarity (Label 1)", color="blue", fill=True, alpha=0.5)
    sns.kdeplot(red_to_red, label="Red-to-Red Similarity (Label 0)", color="red", fill=True, alpha=0.5)
    plt.title("Density Plot of Cosine Similarities within Classes")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Quantify pointwise distance (1 - similarity)
    blue_to_blue_distances = 1 - blue_to_blue
    red_to_red_distances = 1 - red_to_red

    # Summary statistics
    print("Blue-to-Blue Similarities:")
    print(f"Mean: {np.mean(blue_to_blue):.4f}, Std: {np.std(blue_to_blue):.4f}")

    print("\nRed-to-Red Similarities:")
    print(f"Mean: {np.mean(red_to_red):.4f}, Std: {np.std(red_to_red):.4f}")

    print("\nBlue-to-Blue Distances:")
    print(f"Mean: {np.mean(blue_to_blue_distances):.4f}, Std: {np.std(blue_to_blue_distances):.4f}")

    print("\nRed-to-Red Distances:")
    print(f"Mean: {np.mean(red_to_red_distances):.4f}, Std: {np.std(red_to_red_distances):.4f}")


def visualize_embeddings(embeddings, labels=None, title="Embedding Visualization", method="tsne"):
    """
    Visualize sentence embeddings using t-SNE or UMAP.

    Args:
        embeddings (np.ndarray): Sentence embeddings.
        labels (list or np.ndarray): Optional labels for coloring the points.
        title (str): Title of the plot.
        method (str): Dimensionality reduction method, either 'tsne' or 'umap'.
    """

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Reduce dimensionality
    print(f"Reducing dimensionality with {method.upper()}...")
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method.lower() == "umap":
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid method. Choose 'tsne' or 'umap'.")
    
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Plotting
    plt.figure(figsize=(10, 8))

    if labels is not None:
        # Define harmonious colors
        unique_labels = np.unique(labels)
        colors = ['#2E86AB', '#FF6F61', '#6CC644']  # Blue, Coral, and Green

        for i, label in enumerate(unique_labels):
            indices = np.where(labels == label)[0]
            plt.scatter(
                reduced_embeddings[indices, 0],
                reduced_embeddings[indices, 1],
                label=f"Label {label}",
                color=colors[i % len(colors)],
                alpha=0.4,
            )

        plt.legend(title="Labels", loc="best")
    else:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.4)

    plt.title(title)
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.grid(True)
    plt.show()


def visualize_embeddings_with_params(embeddings, labels=None, title="Embedding Visualization", method="tsne", tsne_params=None, ax=None):
    """
    Visualize sentence embeddings using t-SNE or UMAP with configurable hyperparameters.

    Args:
        embeddings (np.ndarray): Sentence embeddings.
        labels (list or np.ndarray): Optional labels for coloring the points.
        title (str): Title of the plot.
        method (str): Dimensionality reduction method, either 'tsne' or 'umap'.
        tsne_params (dict): Optional dictionary of hyperparameters for t-SNE.
                            E.g., {'perplexity': 30, 'learning_rate': 200, 'n_iter': 1000}
        ax (matplotlib.axes.Axes): Optional Axes object for the subplot.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set default t-SNE parameters if not provided
    if tsne_params is None:
        tsne_params = {'perplexity': 30, 'learning_rate': 200, 'n_iter': 250, 'early_exaggeration': 12}

    # Reduce dimensionality
    print(f"Reducing dimensionality with {method.upper()}...")
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42, **tsne_params)
    elif method.lower() == "umap":
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid method. Choose 'tsne' or 'umap'.")
    
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Plotting
    if ax is None:
        ax = plt.gca()

    ax.set_title(title)
    
    if labels is not None:
        # Define harmonious colors
        unique_labels = np.unique(labels)
        colors = ['#2E86AB', '#FF6F61', '#6CC644']  # Blue, Coral, and Green

        for i, label in enumerate(unique_labels):
            indices = np.where(labels == label)[0]
            ax.scatter(
                reduced_embeddings[indices, 0],
                reduced_embeddings[indices, 1],
                label=f"Label {label}",
                color=colors[i % len(colors)],
                alpha=0.4,
            )

        ax.legend(title="Labels", loc="best")
    else:
        ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.4)

    ax.set_xlabel(f"{method.upper()} Dimension 1")
    ax.set_ylabel(f"{method.upper()} Dimension 2")
    ax.grid(True)


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
    #plt.ylim(0, 1)  # Metrics are typically scaled between 0 and 1
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_metrics_with_confidence(metrics_df, x_col="split_size", y_cols=None, title="Model Quality by Training Data Size"):
    """
    Plot mean values of metrics with confidence intervals for each training data split size.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing metrics for each training split size and set.
        x_col (str): Column name for x-axis (e.g., 'split_size').
        y_cols (list): List of column names to plot on the y-axis.
                       Defaults to ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1'].
        title (str): Title for the plot.

    Returns:
        None: Displays the plot.
    """
    if y_cols is None:
        y_cols = ["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"]

    plt.figure(figsize=(15, 8))
    
    # Convert split_size to numeric for sorting and plotting
    metrics_df[x_col] = metrics_df[x_col].str.replace("%", "").astype(float)

    # Sort by split size
    metrics_df = metrics_df.sort_values(x_col)

    for y_col in y_cols:
        # Group by split size to calculate mean and confidence interval
        grouped = metrics_df.groupby(x_col)[y_col]
        mean_values = grouped.mean()
        std_error = grouped.std() / np.sqrt(grouped.count())  # Confidence interval (standard error)
        
        # Plot mean line
        plt.plot(mean_values.index, mean_values, marker="o", label=y_col.replace("_", " ").title())
        
        # Add shaded confidence interval
        plt.fill_between(mean_values.index, mean_values - std_error, mean_values + std_error, alpha=0.2)

    plt.title(title)
    plt.xlabel("Training Data Split Size (%)")
    plt.ylabel("Metric Value")
    plt.ylim(0.5, 1.0)  # Assuming metric values are between 0 and 1
    plt.xticks(metrics_df[x_col].unique())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_predictions_distribution(techniques_predictions):
    techniques = list(techniques_predictions.keys())
    predictions = list(techniques_predictions.values())

    # Assuming binary classification with labels 0 and 1
    # Count how many 0s and 1s each technique predicted
    counts = []
    for preds in predictions:
        unique, counts_ = np.unique(preds, return_counts=True)
        label_count = dict(zip(unique, counts_))
        # Ensure we have both 0 and 1 keys, set to 0 if missing
        label_count_0 = label_count.get(0, 0)
        label_count_1 = label_count.get(1, 0)
        counts.append((label_count_0, label_count_1))

    # Separate the counts for plotting
    counts_0 = [c[0] for c in counts]
    counts_1 = [c[1] for c in counts]

    x = np.arange(len(techniques))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects0 = ax.bar(x - width/2, counts_0, width, label='Label 0')
    rects1 = ax.bar(x + width/2, counts_1, width, label='Label 1')

    ax.set_ylabel('Count of Predicted Labels')
    ax.set_title('Predicted Label Distribution by Weak Labelling Technique')
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, rotation=45, ha='right')
    ax.legend()

    # Optionally, add labels above the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects0)
    autolabel(rects1)

    plt.tight_layout()
    plt.show()
