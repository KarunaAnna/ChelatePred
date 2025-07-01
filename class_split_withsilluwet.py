import os
import re
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = "results/plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def extract_all_features_from_gml(gml_path):
    G = nx.read_gml(gml_path)
    graph_features = G.graph.copy()
    meta_data = graph_features.pop("meta_data", {})
    for k, v in meta_data.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                graph_features[f"{k}_{sub_k}"] = sub_v
        else:
            graph_features[f"meta_{k}"] = v
    flat_graph_features = {
        f"graph_{k}": v for k, v in graph_features.items()
        if not isinstance(v, (list, dict))
    }

    node_data = [
        [v for v in d.values() if isinstance(v, (int, float))]
        for _, d in G.nodes(data=True)
    ]
    node_array = np.array(node_data, dtype=np.float64) if node_data else np.empty((0,))
    node_agg = {}
    if node_array.size > 0:
        for i, (mean, std) in enumerate(zip(np.nanmean(node_array, axis=0), np.nanstd(node_array, axis=0))):
            node_agg[f"node_feat_{i}_mean"] = mean
            node_agg[f"node_feat_{i}_std"] = std

    edge_data = [
        [v for v in d.values() if isinstance(v, (int, float))]
        for _, _, d in G.edges(data=True)
    ]
    edge_array = np.array(edge_data, dtype=np.float64) if edge_data else np.empty((0,))
    edge_agg = {}
    if edge_array.size > 0:
        for i, (mean, std) in enumerate(zip(np.nanmean(edge_array, axis=0), np.nanstd(edge_array, axis=0))):
            edge_agg[f"edge_feat_{i}_mean"] = mean
            edge_agg[f"edge_feat_{i}_std"] = std

    all_features = {**flat_graph_features, **node_agg, **edge_agg}
    return all_features

def build_dataset_from_excel(excel_path, gml_folder, filename_col="GML File", label_col="Metal Ion"):
    df_input = pd.read_excel(excel_path)
    X, y, file_names = [], [], []

    for _, row in df_input.iterrows():
        gml_file = row[filename_col]
        label = row[label_col]
        gml_path = os.path.join(gml_folder, gml_file)

        try:
            features = extract_all_features_from_gml(gml_path)
            X.append(features)
            y.append(label)
            file_names.append(gml_file)
        except Exception as e:
            print(f"Skipping {gml_file}: {e}")

    df_X = pd.DataFrame(X)
    df_X["Label"] = y
    df_X["GML File"] = file_names

    allowed_classes = {"Co", "Cu", "Ni", "Mn", "Fe", "Zn"}
    df_X["Label"] = df_X["Label"].apply(lambda x: x if x in allowed_classes else "others")

    return df_X

def plot_feature_importance(importances, feature_names, classifier_name, top_n=20):
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[sorted_idx]
    top_feature_names = np.array(feature_names)[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_importances)), top_importances[::-1])
    plt.yticks(range(len(top_feature_names)), top_feature_names[::-1])
    plt.xlabel("Importance Score")
    plt.title(f"Top {top_n} Feature Importances - {classifier_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"feature_importance_{classifier_name}.png"))
    plt.close()

def train_and_evaluate(df_features):
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    non_feature_cols = ["Label", "GML File"]
    X_raw = df_features.drop(columns=[
        col for col in df_features.columns
        if col in non_feature_cols
        or df_features[col].dtype == 'object'
        or re.match(r"(node_feat_0_.*|graph_meta_metal_center_element|graph_meta_metal_center_group|graph_meta_metal_center_period)", col)
    ])
    y = df_features["Label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # === Impute missing values before clustering ===
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X_raw)

    # === Cluster-Aware Train/Test Split ===
    # === Determine optimal number of clusters using silhouette score ===
    best_score = -1
    best_k = 2
    for k in range(2, 10):
        kmeans_k = KMeans(n_clusters=k, random_state=42)
        labels_k = kmeans_k.fit_predict(X)
        score = silhouette_score(X, labels_k)
        if score > best_score:
            best_score = score
            best_k = k
    print(f"✅ Optimal number of clusters based on silhouette score: {best_k}")

    # Use best_k for actual cluster-based split
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    test_cluster = best_k - 1  # last cluster as test
    test_mask = cluster_labels == test_cluster
    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y_encoded[train_mask], y_encoded[test_mask]

    # === PCA + t-SNE Visualization ===
    X_combined = np.vstack([X_train, X_test])
    X_pca = PCA(n_components=2).fit_transform(X_combined)
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_combined)

    split_labels = ['Train'] * len(X_train) + ['Test'] * len(X_test)
    split_colors = ['blue' if s == 'Train' else 'red' for s in split_labels]

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=split_colors, alpha=0.6)
    plt.title("PCA of Feature Space (Cluster-Based Train/Test)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(["Train", "Test"])
    plt.tight_layout()
    plt.savefig("outputs/feature_space_pca_cluster_split.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=split_colors, alpha=0.6)
    plt.title("t-SNE of Feature Space (Cluster-Based Train/Test)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(["Train", "Test"])
    plt.tight_layout()
    plt.savefig("outputs/feature_space_tsne_cluster_split.png")
    plt.close()

    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        "SVM": SVC(probability=True),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
    }

    for name, model in classifiers.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"\n=== Classification Report: {name} ===")
        report_str = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        print(report_str)

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"outputs/{name.replace(' ', '_')}_confusion_matrix.png")
        plt.close()

        if hasattr(model, "feature_importances_"):
            plot_feature_importance(
                model.feature_importances_,
                feature_names=X_raw.columns,
                classifier_name=name.replace(" ", "_"),
                top_n=20
            )
            print(f"✅ Saved feature importance plot for {name}")
        else:
            print(f"⚠️ {name} does not support feature_importances_")

        with open(f"outputs/{name.replace(' ', '_')}_classification_report.txt", "w") as f:
            f.write(f"Classification Report for {name}\n{report_str}")

def main():
    excel_path = "./metal_complexDB.xlsx"
    gml_folder = "./gml_files"

    df_all = build_dataset_from_excel(excel_path, gml_folder)

    if df_all.empty:
        raise ValueError("Dataset is empty. Check your Excel or GML file content.")
    else:
        print(f"✅ Loaded {df_all.shape[0]} samples with {df_all.shape[1]} features.")
        df_all.to_csv("outputs/feature_space.csv", index=False)
        train_and_evaluate(df_all)

if __name__ == "__main__":
    main()

