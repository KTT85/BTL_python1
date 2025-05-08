import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load and prepare data with mixed features for all roles including goalkeepers
def load_and_prepare_data(file_path='results.csv'):
    df = pd.read_csv(file_path)
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # Selected hybrid feature set
    selected_columns = [
    'Goals', 'xG', 'GCA', 'SCA', 'Assists',
    'SoT%', 'SoT/90', 'G/Sh',
    'Cmp%', 'KP', 'PrgP', 'PrgC',
    'Passes 1/3', 'PPA', 'CrsPA',
    'Tkl', 'TklW', 'Int', 'Blocks', 'Sh',
    'Save%', 'CS%', 'GA90', 'PKsv%',
    'Touches', 'Lost', 'Won%', 'Recov'
]

    # Handle missing data: fill goalkeeper stats with 0 for outfield players and vice versa
    df_features = df[selected_columns].replace("N/a", np.nan).fillna(0)
    df_features = df_features.astype(float)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)

    return df, df_features, scaled_features

# Determine best number of clusters
def determine_best_k(data, k_range=range(2, 10)):
    scores = []
    valid_k = []
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(data)
            if len(set(labels)) <= 1:
                continue
            score = silhouette_score(data, labels)
            scores.append(score)
            valid_k.append(k)
        except Exception as e:
            logging.warning(f"Silhouette score failed for k={k}: {e}")
            continue

    if not scores:
        raise ValueError("No valid k values found for silhouette score.")

    best_k = valid_k[np.argmax(scores)]
    return best_k, scores

# Plot PCA clusters
def plot_clusters(data, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    fig = plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', s=30)
    plt.title('K-means Clustering with PCA (2D)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    return fig

# Plot silhouette scores
def plot_silhouette_scores(scores, k_range):
    fig = plt.figure()
    plt.plot(k_range, scores, marker='o')
    plt.title('Silhouette Scores for k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    return fig

# Save figures to PDF
def save_plots_to_pdf(figures, filename='classify.pdf'):
    with PdfPages(filename) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
    logging.info(f"Saved plots to {filename}")

# Main function
def main():
    try:
        df, df_features, scaled_features = load_and_prepare_data()
        k_range = range(2, 10)
        best_k, silhouette_scores = determine_best_k(scaled_features, k_range)

        logging.info(f"Best number of clusters (k): {best_k}")

        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(scaled_features)

        fig1 = plot_clusters(scaled_features, cluster_labels)
        fig2 = plot_silhouette_scores(silhouette_scores, k_range)

        save_plots_to_pdf([fig1, fig2])

    except Exception as e:
        logging.error(f"Lỗi xảy ra: {e}")

# Run the script
if __name__ == '__main__':
    main()
