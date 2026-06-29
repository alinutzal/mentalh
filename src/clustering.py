from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/reddit_classified.csv")
DEFAULT_OUTPUT = Path("results/umap_sentence_transformer_topics.png")
DEFAULT_COORDS = Path("results/umap_sentence_transformer_coords.csv")
DEFAULT_CLUSTER_OUTPUT = Path("results/umap_all_true_centroid_clusters.png")
DEFAULT_COLOR_COLS = ["college", "gaming", "mental_health"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed mental-health text with SentenceTransformer, run UMAP, and plot it."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV input file.")
    parser.add_argument("--text-col", default="text", help="Column containing text to embed.")
    parser.add_argument(
        "--color-cols",
        nargs="+",
        default=DEFAULT_COLOR_COLS,
        help="Columns used to color separate UMAP plots.",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional number of rows to sample before embedding.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Plot output path.")
    parser.add_argument(
        "--cluster-output",
        type=Path,
        default=DEFAULT_CLUSTER_OUTPUT,
        help="Plot output path for all-true centroid clustering.",
    )
    parser.add_argument(
        "--kmeans-max-iter",
        type=int,
        default=100,
        help="Maximum KMeans iterations when seeded by all-true UMAP points.",
    )
    parser.add_argument(
        "--coords-output",
        type=Path,
        default=DEFAULT_COORDS,
        help="CSV output path for UMAP coordinates.",
    )
    return parser.parse_args()


def load_data(
    path: Path,
    text_col: str,
    color_cols: list[str],
    sample: int | None,
    seed: int,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [col for col in [text_col, *color_cols] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) in {path}: {', '.join(missing)}")

    keep_cols = [text_col, *[col for col in df.columns if col != text_col]]
    df = df[keep_cols].dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].reset_index(drop=True)

    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)

    return df


def as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def all_true_mask(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return pd.concat([as_bool_series(df[col]) for col in cols], axis=1).all(axis=1)


def embed_texts(texts: list[str], model_name: str, batch_size: int):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def run_umap(embeddings, neighbors: int, min_dist: float, seed: int):
    try:
        import umap
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: umap-learn. Install it with: pip install umap-learn"
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def run_centroid_clustering(
    df: pd.DataFrame,
    centroid_cols: list[str],
    seed: int,
    max_iter: int,
) -> pd.DataFrame:
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: scikit-learn. Install it with: pip install scikit-learn"
        ) from exc

    centroid_mask = all_true_mask(df, centroid_cols)
    initial_centroids = df.loc[centroid_mask, ["umap_x", "umap_y"]].to_numpy()
    if len(initial_centroids) == 0:
        raise ValueError(f"No rows have all centroid columns true: {', '.join(centroid_cols)}")

    points = df[["umap_x", "umap_y"]].to_numpy()
    kmeans = KMeans(
        n_clusters=len(initial_centroids),
        init=initial_centroids,
        n_init=1,
        max_iter=max_iter,
        random_state=seed,
    )
    labels = kmeans.fit_predict(points)

    df = df.copy()
    df["all_topic_centroid"] = centroid_mask
    df["centroid_cluster"] = labels
    df["centroid_distance"] = kmeans.transform(points).min(axis=1)
    df["centroid_cluster_size"] = df.groupby("centroid_cluster")["centroid_cluster"].transform("size")
    return df


def plot_umap(df: pd.DataFrame, color_cols: list[str], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install it with: pip install matplotlib"
        ) from exc

    fig, axes = plt.subplots(1, len(color_cols), figsize=(6 * len(color_cols), 5.5), sharex=True, sharey=True)
    if len(color_cols) == 1:
        axes = [axes]

    false_color = "#b8bcc6"
    true_color = "#d84a3a"

    for ax, col in zip(axes, color_cols):
        values = as_bool_series(df[col])
        false_mask = ~values
        true_mask = values

        ax.scatter(
            df.loc[false_mask, "umap_x"],
            df.loc[false_mask, "umap_y"],
            s=6,
            alpha=0.18,
            color=false_color,
            linewidths=0,
        )
        ax.scatter(
            df.loc[true_mask, "umap_x"],
            df.loc[true_mask, "umap_y"],
            s=8,
            alpha=0.75,
            color=true_color,
            linewidths=0,
        )
        ax.set_title(f"{col} ({true_mask.sum():,} true)")
        ax.set_xlabel("UMAP 1")
        ax.grid(alpha=0.12)

    axes[0].set_ylabel("UMAP 2")
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=true_color, markersize=7, label="True"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=false_color, markersize=7, label="False"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("SentenceTransformer UMAP Colored by Topic Flags", y=1.02)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_centroid_clusters(df: pd.DataFrame, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install it with: pip install matplotlib"
        ) from exc

    centroid_mask = as_bool_series(df["all_topic_centroid"])
    cluster_count = df["centroid_cluster"].nunique()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    axes[0].scatter(
        df.loc[~centroid_mask, "umap_x"],
        df.loc[~centroid_mask, "umap_y"],
        s=6,
        alpha=0.18,
        color="#b8bcc6",
        linewidths=0,
    )
    axes[0].scatter(
        df.loc[centroid_mask, "umap_x"],
        df.loc[centroid_mask, "umap_y"],
        s=26,
        alpha=0.9,
        color="#d84a3a",
        edgecolors="#111111",
        linewidths=0.35,
    )
    axes[0].set_title(f"All Three Flags True ({centroid_mask.sum():,} centroids)")

    axes[1].scatter(
        df["umap_x"],
        df["umap_y"],
        c=df["centroid_cluster"],
        cmap="turbo",
        s=6,
        alpha=0.45,
        linewidths=0,
    )
    axes[1].scatter(
        df.loc[centroid_mask, "umap_x"],
        df.loc[centroid_mask, "umap_y"],
        s=22,
        color="#111111",
        alpha=0.9,
        linewidths=0,
    )
    axes[1].set_title(f"KMeans Seeded by Those Centroids ({cluster_count:,} clusters)")

    for ax in axes:
        ax.set_xlabel("UMAP 1")
        ax.grid(alpha=0.12)
    axes[0].set_ylabel("UMAP 2")

    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#d84a3a", markeredgecolor="#111111", markersize=7, label="college + gaming + mental_health"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#111111", markersize=7, label="seed centroid"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    fig.suptitle("UMAP with All-True Points Used as KMeans Centroids", y=1.02)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_data(args.input, args.text_col, args.color_cols, args.sample, args.seed)

    print(f"Loaded {len(df):,} rows from {args.input}")
    embeddings = embed_texts(df[args.text_col].tolist(), args.model, args.batch_size)

    print("Running UMAP...")
    coords = run_umap(embeddings, args.neighbors, args.min_dist, args.seed)
    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]

    print("Running centroid-seeded clustering...")
    df = run_centroid_clustering(df, args.color_cols, args.seed, args.kmeans_max_iter)

    args.coords_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.coords_output, index=False)
    plot_umap(df, args.color_cols, args.output)
    plot_centroid_clusters(df, args.cluster_output)

    print(f"Saved plot to {args.output}")
    print(f"Saved centroid cluster plot to {args.cluster_output}")
    print(f"Saved coordinates to {args.coords_output}")


if __name__ == "__main__":
    main()
