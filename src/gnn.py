from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


DEFAULT_INPUT = Path("data/reddit_classified.csv")
DEFAULT_EMBEDDINGS = Path("results/gnn_sentence_embeddings.npy")
DEFAULT_OUTPUT = Path("results/gnn_predictions.csv")
RELATION_COLS = ["college", "gaming", "mental_health"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-relational GNN over Reddit posts with topic-typed edges."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV.")
    parser.add_argument("--text-col", default="text", help="Text column used for node features.")
    parser.add_argument(
        "--target",
        choices=["status", "mental_health", "intersectional"],
        default="status",
        help="Node-classification target.",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for node features.",
    )
    parser.add_argument(
        "--embeddings-cache",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
        help="Path for cached sentence-transformer embeddings.",
    )
    parser.add_argument("--sample", type=int, default=None, help="Optional row sample for faster runs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--neighbors",
        type=int,
        default=8,
        help="Nearest shared-topic neighbors per node for each edge type.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout probability.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test split size.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation split from train data.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Prediction CSV output.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def load_posts(path: Path, text_col: str, sample: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [text_col, "status", *RELATION_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) in {path}: {', '.join(missing)}")

    df = df.dropna(subset=[text_col, "status"]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].reset_index(drop=True)

    for col in RELATION_COLS:
        df[col] = as_bool_series(df[col])

    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)

    return df


def get_embeddings(
    texts: list[str],
    model_name: str,
    cache_path: Path,
    seed: int,
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached) == len(texts):
            print(f"Loaded cached embeddings from {cache_path}")
            return cached.astype(np.float32)
        print(f"Ignoring cache with {len(cached):,} rows; current data has {len(texts):,} rows.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    torch.manual_seed(seed)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"Saved embeddings to {cache_path}")
    return embeddings


def build_typed_edges(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    relation_cols: list[str],
    neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    edges: list[np.ndarray] = []
    edge_types: list[np.ndarray] = []
    relation_to_id = {name: idx for idx, name in enumerate(relation_cols)}

    for relation, relation_id in relation_to_id.items():
        node_ids = np.flatnonzero(df[relation].to_numpy(dtype=bool))
        if len(node_ids) <= 1:
            print(f"Skipping relation {relation}: only {len(node_ids)} positive node(s).")
            continue

        k = min(neighbors + 1, len(node_ids))
        nn_index = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn_index.fit(embeddings[node_ids])
        neighbor_positions = nn_index.kneighbors(return_distance=False)

        src_local = np.repeat(np.arange(len(node_ids)), k - 1)
        dst_local = neighbor_positions[:, 1:].reshape(-1)
        src = node_ids[src_local]
        dst = node_ids[dst_local]

        relation_edges = np.column_stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
        relation_edges = np.unique(relation_edges, axis=0)
        edges.append(relation_edges)
        edge_types.append(np.full(len(relation_edges), relation_id, dtype=np.int64))
        print(f"{relation}: {len(node_ids):,} positive nodes, {len(relation_edges):,} directed edges")

    if not edges:
        raise ValueError("No graph edges were created. Try a larger sample or fewer filters.")

    edge_index = torch.tensor(np.vstack(edges).T, dtype=torch.long)
    edge_type = torch.tensor(np.concatenate(edge_types), dtype=torch.long)
    return edge_index, edge_type, relation_to_id


def build_labels(df: pd.DataFrame, target: str) -> tuple[np.ndarray, list[str]]:
    if target == "status":
        encoder = LabelEncoder()
        labels = encoder.fit_transform(df["status"].astype(str))
        return labels.astype(np.int64), list(encoder.classes_)

    if target == "mental_health":
        labels = df["mental_health"].to_numpy(dtype=np.int64)
        return labels, ["false", "true"]

    labels = df[RELATION_COLS].all(axis=1).to_numpy(dtype=np.int64)
    return labels, ["not_all_three", "all_three"]


def stratified_masks(labels: np.ndarray, test_size: float, val_size: float, seed: int):
    indices = np.arange(len(labels))
    stratify = labels if pd.Series(labels).value_counts().min() >= 2 else None
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    train_val_labels = labels[train_val_idx]
    stratify_train = train_val_labels if pd.Series(train_val_labels).value_counts().min() >= 2 else None
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=seed,
        stratify=stratify_train,
    )

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )


def class_weights(labels: np.ndarray, train_idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    train_labels = labels[train_idx.numpy()]
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


class RelGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.rel_linears = nn.ModuleList(nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        out = self.self_linear(x)
        num_nodes = x.size(0)

        for relation_id, linear in enumerate(self.rel_linears):
            mask = edge_type == relation_id
            if not torch.any(mask):
                continue

            src = edge_index[0, mask]
            dst = edge_index[1, mask]
            messages = linear(x[src])
            aggregated = torch.zeros(num_nodes, messages.size(1), device=x.device)
            aggregated.index_add_(0, dst, messages)

            degree = torch.zeros(num_nodes, device=x.device)
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
            out = out + aggregated / degree.clamp(min=1).unsqueeze(1)

        return out


class RelationalGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_relations: int, dropout: float):
        super().__init__()
        self.conv1 = RelGraphConv(in_dim, hidden_dim, num_relations)
        self.conv2 = RelGraphConv(hidden_dim, out_dim, num_relations)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index, edge_type)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    y: torch.Tensor,
    idx: torch.Tensor,
) -> float:
    model.eval()
    logits = model(x, edge_index, edge_type)
    pred = logits[idx].argmax(dim=1)
    return (pred == y[idx]).float().mean().item()


def train_model(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    num_classes: int,
    args: argparse.Namespace,
) -> RelationalGNN:
    model = RelationalGNN(
        in_dim=x.size(1),
        hidden_dim=args.hidden_dim,
        out_dim=num_classes,
        num_relations=len(RELATION_COLS),
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = class_weights(y.numpy(), train_idx, num_classes)

    best_state = None
    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index, edge_type)
        loss = F.cross_entropy(logits[train_idx], y[train_idx], weight=weights)
        loss.backward()
        optimizer.step()

        val_acc = evaluate(model, x, edge_index, edge_type, y, val_idx)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        print(f"epoch={epoch:03d} loss={loss.item():.4f} val_acc={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def save_predictions(
    df: pd.DataFrame,
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    class_names: list[str],
    output: Path,
) -> None:
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(x, edge_index, edge_type), dim=1)
        pred_ids = probs.argmax(dim=1).cpu().numpy()

    result = df.copy()
    result["gnn_prediction"] = [class_names[idx] for idx in pred_ids]
    result["gnn_confidence"] = probs.max(dim=1).values.cpu().numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.target == "mental_health":
        print("Note: target is mental_health and one edge type is also mental_health.")
        print("For a leakage check, rerun with --color/edge logic modified to remove that relation.")

    df = load_posts(args.input, args.text_col, args.sample, args.seed)
    print(f"Loaded {len(df):,} posts from {args.input}")

    embeddings = get_embeddings(
        texts=df[args.text_col].tolist(),
        model_name=args.model,
        cache_path=args.embeddings_cache,
        seed=args.seed,
    )
    edge_index, edge_type, relation_to_id = build_typed_edges(df, embeddings, RELATION_COLS, args.neighbors)
    print(f"Relation ids: {relation_to_id}")
    print(f"Graph: {len(df):,} nodes, {edge_index.size(1):,} directed typed edges")

    labels, class_names = build_labels(df, args.target)
    print("Classes:", dict(zip(class_names, np.bincount(labels, minlength=len(class_names)).tolist())))

    train_idx, val_idx, test_idx = stratified_masks(labels, args.test_size, args.val_size, args.seed)
    x = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    model = train_model(x, edge_index, edge_type, y, train_idx, val_idx, len(class_names), args)

    test_acc = evaluate(model, x, edge_index, edge_type, y, test_idx)
    print(f"test_acc={test_acc:.4f}")

    with torch.no_grad():
        logits = model(x, edge_index, edge_type)
        y_pred = logits[test_idx].argmax(dim=1).cpu().numpy()

    print(
        classification_report(
            y[test_idx].cpu().numpy(),
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    save_predictions(df, model, x, edge_index, edge_type, class_names, args.output)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
