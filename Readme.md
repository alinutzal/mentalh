# Mental Health Reddit Graph Analysis

This project explores Reddit mental-health posts with two complementary approaches:

- UMAP visualization of sentence-transformer text embeddings.
- A multi-relational GNN where Reddit posts are nodes and topic-specific connections are typed edges.

The main dataset expected by the current scripts is:

```text
data/reddit_classified.csv
```

Expected columns:

```text
Unique_ID,text,status,college,gaming,mental_health
```

## Setup

Create or activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## UMAP Clustering

`clustering.py` embeds each post with a sentence-transformer model, reduces the embeddings to 2D with UMAP, and plots topic flags.

Run the full workflow:

```powershell
python clustering.py
```

Run a faster sample:

```powershell
python clustering.py --sample 2000
```

Main outputs:

```text
results/umap_sentence_transformer_topics.png
results/umap_sentence_transformer_coords.csv
results/umap_all_true_centroid_clusters.png
```

The centroid-clustering plot highlights posts where all three flags are true:

```text
college == True
gaming == True
mental_health == True
```

Those all-true UMAP points are used as initial KMeans centroids. The coordinates CSV includes:

```text
all_topic_centroid
centroid_cluster
centroid_distance
centroid_cluster_size
```

## GNN Node Classification

`gnn.py` builds a multi-relational graph over Reddit posts.

Graph setup:

- Nodes: Reddit posts
- Node features: sentence-transformer embeddings of `text`
- Edge types: `college`, `gaming`, `mental_health`
- Edges: semantic nearest-neighbor links among posts that share the same topic flag
- Model: pure PyTorch relational GCN-style network
- Task: node classification

Default task: predict `status`.

```powershell
python gnn.py
```

Available targets:

```powershell
python gnn.py --target status
python gnn.py --target mental_health
python gnn.py --target intersectional
```

`intersectional` means:

```text
college == True and gaming == True and mental_health == True
```

Useful fast test command:

```powershell
python gnn.py --sample 500 --epochs 2 --neighbors 3
```

Main output:

```text
results/gnn_predictions.csv
```

The script also caches sentence-transformer embeddings by default:

```text
results/gnn_sentence_embeddings.npy
```

## Research Questions

Core GNN questions:

1. Can a multi-relational GNN improve mental-health classification compared with text-only models?
2. Do `college`, `gaming`, and `mental_health` edge types contribute differently to node classification performance?
3. Which relation type is most useful for predicting mental-health status?
4. Does combining all three relation types improve classification more than using any single relation type alone?
5. Can graph structure recover weak or hidden mental-health signals that are not obvious from text embeddings alone?

Intersectional questions:

1. Do posts at the intersection of college, gaming, and mental health form a distinct graph community?
2. Are college-gaming posts structurally closer to anxiety, depression, or suicidal posts?
3. Are gaming-related mental-health posts more isolated, more clustered, or more bridge-like than other mental-health posts?
4. Do all-three-true posts act as bridges between academic stress, gaming, and broader mental-health communities?

Methods questions:

1. Does a relational GCN outperform a standard GCN that ignores edge types?
2. How much does each edge type contribute when removed in an ablation study?
3. Does graph-based classification help with rare or high-risk labels such as suicidal ideation?
4. Can semantic nearest-neighbor edges plus topic-specific edge types identify posts missed by keyword or classifier flags?

## Notes

If `--target mental_health` is used, remember that one edge type is also `mental_health`. That can be useful for graph propagation experiments, but for a stricter leakage check, remove the `mental_health` relation and compare performance.
