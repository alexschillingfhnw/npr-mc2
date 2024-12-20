import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Functions for weak labeling techniques
def majority_vote(similarity_matrix, labels_labeled, k=5):
    """
    Assign weak labels using majority vote among top-k most similar sentences.
    """
    weak_labels = []
    for similarities in similarity_matrix:
        top_k_indices = np.argsort(similarities)[-k:]
        top_k_labels = labels_labeled[top_k_indices]
        label_counts = Counter(top_k_labels)
        weak_labels.append(label_counts.most_common(1)[0][0])
    return np.array(weak_labels)


def weighted_vote(similarity_matrix, labels_labeled, k=5):
    """
    Assign weak labels using weighted vote based on cosine similarity scores.
    """
    weak_labels = []
    for similarities in similarity_matrix:
        top_k_indices = np.argsort(similarities)[-k:]
        top_k_labels = labels_labeled[top_k_indices]
        top_k_weights = similarities[top_k_indices]
        
        weighted_counts = Counter()
        for label, weight in zip(top_k_labels, top_k_weights):
            weighted_counts[label] += weight
        weak_labels.append(weighted_counts.most_common(1)[0][0])
    return np.array(weak_labels)


def centroid_based_labeling(embeddings_unlabeled, centroids):
    """
    Assign weak labels based on closest centroid in embedding space.
    """
    weak_labels = []
    for embedding in embeddings_unlabeled:
        distances = [np.linalg.norm(embedding - centroid) for centroid in centroids]
        weak_labels.append(np.argmin(distances))  # Label corresponds to closest centroid
    return np.array(weak_labels)


def compute_class_centroids(embeddings_labeled, labels_labeled):
    """
    Compute centroids for each class in the embedding space.
    """
    unique_labels = np.unique(labels_labeled)
    centroids = []
    for label in unique_labels:
        class_embeddings = embeddings_labeled[labels_labeled == label]
        centroid = np.median(class_embeddings, axis=0)
        centroids.append(centroid)
    return centroids


def evaluate_and_compare_techniques(embeddings_labeled, labels_labeled, validation_labels, embeddings_validation, k=5):
    """
    Apply and evaluate multiple weak labeling techniques using the validation set with ground truth.
    """
    # Compute similarity matrix for the validation set
    similarity_matrix_validation = cosine_similarity(embeddings_validation, embeddings_labeled)
    
    # Precompute class centroids
    centroids = compute_class_centroids(embeddings_labeled, labels_labeled)
    
    # Apply different techniques on validation set
    techniques = {
        "Majority Vote": majority_vote(similarity_matrix_validation, labels_labeled, k),
        "Weighted Vote": weighted_vote(similarity_matrix_validation, labels_labeled, k),
        "Centroid-Based": centroid_based_labeling(embeddings_validation, centroids),
    }
    
    # Evaluate techniques using validation ground truth
    results = {}
    for technique_name, weak_labels in techniques.items():
        precision, recall, f1, _ = precision_recall_fscore_support(validation_labels, weak_labels, average="binary")
        accuracy = accuracy_score(validation_labels, weak_labels)
        results[technique_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    return results, techniques


def apply_best_technique_to_test(embeddings_labeled, labels_labeled, embeddings_test, best_technique, k=5):
    """
    Apply the best weak labeling technique to the test set.
    """
    similarity_matrix_test = cosine_similarity(embeddings_test, embeddings_labeled)
    
    if best_technique == "Majority Vote":
        return majority_vote(similarity_matrix_test, labels_labeled, k)
    elif best_technique == "Weighted Vote":
        return weighted_vote(similarity_matrix_test, labels_labeled, k)
    elif best_technique == "Centroid-Based":
        centroids = compute_class_centroids(embeddings_labeled, labels_labeled)
        return centroid_based_labeling(embeddings_test, centroids)
    else:
        raise ValueError(f"Unknown technique: {best_technique}")


def weighted_vote_with_confidence(similarity_matrix, labels_labeled, k=5):
    """
    Perform weighted voting and return both the predicted labels and a confidence score.
    Confidence is defined as the margin between the top label's total similarity and the second top label.
    """
    weak_labels = []
    confidences = []
    
    for similarities in similarity_matrix:
        top_k_indices = np.argsort(similarities)[-k:]
        top_k_labels = labels_labeled[top_k_indices]
        top_k_weights = similarities[top_k_indices]

        # Aggregate weights by label
        weighted_counts = Counter()
        for label, weight in zip(top_k_labels, top_k_weights):
            weighted_counts[label] += weight
        
        # Sort labels by total weight
        label_weights = weighted_counts.most_common()
        predicted_label = label_weights[0][0]
        
        # Confidence: difference between top label and second label weights
        if len(label_weights) > 1:
            confidence = label_weights[0][1] - label_weights[1][1]
        else:
            # If only one label present (rare if k>1), confidence is just the total weight
            confidence = label_weights[0][1]

        weak_labels.append(predicted_label)
        confidences.append(confidence)

    return np.array(weak_labels), np.array(confidences)
