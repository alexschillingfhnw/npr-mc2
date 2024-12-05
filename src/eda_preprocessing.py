import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.utils import resample
import pandas as pd

def explore_class_distribution(data, dataset_name):
    """Explore class distribution."""
    class_counts = Counter(data['label'])
    print(f"\nClass Distribution in {dataset_name}:")
    for label, count in class_counts.items():
        print(f"Label {label}: {count} samples")

    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title(f"Class Distribution in {dataset_name}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

def basic_statistics(data, dataset_name):
    """Perform basic statistics on sentence lengths."""
    sentence_lengths = data['sentence'].apply(len)
    print(f"\nBasic Statistics for {dataset_name}:")
    print(f"Average sentence length: {sentence_lengths.mean():.2f}")
    print(f"Max sentence length: {sentence_lengths.max()}")
    print(f"Min sentence length: {sentence_lengths.min()}")

    plt.figure(figsize=(8, 5))
    sns.histplot(sentence_lengths, kde=True, bins=30)
    plt.title(f"Sentence Length Distribution in {dataset_name}")
    plt.xlabel("Sentence Length")
    plt.ylabel("Frequency")
    plt.show()

def plot_word_length_distribution(data, dataset_name):
    """Plot the distribution of word lengths in sentences."""
    word_lengths = data['sentence'].apply(lambda x: len(x.split()))
    print(f"\nWord Length Statistics for {dataset_name}:")
    print(f"Average word count: {word_lengths.mean():.2f}")
    print(f"Max word count: {word_lengths.max()}")
    print(f"Min word count: {word_lengths.min()}")
    
    plt.figure(figsize=(8, 5))
    sns.histplot(word_lengths, kde=True, bins=30)
    plt.title(f"Word Length Distribution in {dataset_name}")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.show()

def plot_token_length_distribution(token_lengths, dataset_name):
    """Plot token length distribution."""
    plt.figure(figsize=(8, 5))
    sns.histplot(token_lengths, kde=True, bins=30)
    plt.title(f"Token Length Distribution in {dataset_name}")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.show()

def calculate_token_lengths(data, tokenizer, dataset_name):
    """Calculate token lengths for the dataset."""
    token_lengths = [
        len(tokenizer.encode(sentence, truncation=False, add_special_tokens=True))
        for sentence in data['sentence']
    ]
    print(f"\nToken Length Statistics for {dataset_name}:")
    print(f"Average token length: {sum(token_lengths) / len(token_lengths):.2f}")
    print(f"Max token length: {max(token_lengths)}")
    print(f"Min token length: {min(token_lengths)}")
    return token_lengths

def tokenize_data(tokenizer, data):
    """Tokenizes text data using a Hugging Face tokenizer."""
    return tokenizer(
        data["sentence"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=128,  # Adjust as per model requirements
        return_tensors="pt"
    )

def create_nested_splits(data, sizes, random_state=42):
    """Create hierarchically nested training datasets of varying sizes."""
    nested_splits = {}
    data_shuffled = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    for size in sizes:
        sample_size = int(len(data) * size)
        nested_splits[f"{int(size * 100)}%"] = data_shuffled.iloc[:sample_size]
        print(f"Created split: {int(size * 100)}% with {sample_size} samples.")
    
    return nested_splits

def balance_dataset_undersample(data, label_col="label"):
    """
    Balances the dataset by undersampling the majority class.
    """
    # Separate majority and minority classes
    majority_class = data[data[label_col] == data[label_col].value_counts().idxmax()]
    minority_class = data[data[label_col] == data[label_col].value_counts().idxmin()]

    # Undersample majority class
    majority_class_downsampled = resample(
        majority_class,
        replace=False,  # No replacement
        n_samples=len(minority_class),  # Match minority class size
        random_state=42  # Reproducibility
    )

    # Combine minority class with undersampled majority class
    balanced_data = pd.concat([minority_class, majority_class_downsampled])

    print(f"Class distribution after undersampling:\n{balanced_data[label_col].value_counts()}")
    return balanced_data
