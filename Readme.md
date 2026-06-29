# Mental Health Reddit Graph Analysis

This project explores Reddit mental-health discussions with complementary graph and embedding approaches:

- UMAP visualization of sentence-transformer text embeddings.
- A multi-relational GNN where Reddit posts are nodes and topic-specific connections are typed edges.
- An author-level graph design for unlabeled data, where users are nodes and the goal is community discovery or self-supervised representation learning.

The main dataset expected by the current scripts is:

```text
data/reddit_classified.csv
```

Expected columns:

```text
Unique_ID,text,status,college,gaming,mental_health
```

For the author-node setup, an author/user column is also needed, for example:

```text
author,text,college,gaming,mental_health
```

In that version, `status` is optional because the analysis can be unsupervised or self-supervised.

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

## Author-Node Graph Without Labels

If there are no `status` labels, the project can shift from supervised post classification to unsupervised or self-supervised author analysis.

Author-node setup:

- Nodes: Reddit authors/users
- Node features: aggregated sentence-transformer embeddings from each author's posts
- Edge types: `college`, `gaming`, `mental_health`
- Edges: authors are connected when they post about similar topics, share semantic similarity, or participate in the same topic-defined neighborhoods
- Task: community discovery, representation learning, link prediction, or weak-risk scoring

Useful author features:

- Mean embedding across all posts by the author
- Separate mean embeddings for college, gaming, and mental-health posts
- Counts or rates of posts with each topic flag
- Fraction of posts where all three flags are true
- Optional temporal features, such as whether mental-health language increases over time

Possible edge definitions:

- `college` edge: two authors have semantically similar college-related posts
- `gaming` edge: two authors have semantically similar gaming-related posts
- `mental_health` edge: two authors have semantically similar mental-health posts
- Intersection edge: two authors both discuss college, gaming, and mental health

Without `status`, the model should not be framed as supervised mental-health diagnosis. Better tasks are:

- Community detection: find groups of authors with similar experiences
- Link prediction: predict missing topic-specific connections between authors
- Contrastive learning: learn author embeddings by pulling connected authors closer and pushing unrelated authors apart
- Anomaly or bridge detection: identify authors who connect college, gaming, and mental-health communities
- Weak supervision: create approximate labels from topic flags or keyword rules, then validate them manually

Example research question:

> Without diagnostic labels, can a multi-relational author graph reveal communities of users whose posts connect college life, gaming, and mental-health distress?

This author-node framing is useful when the research goal is discovery rather than prediction. It can show how users cluster, which authors bridge communities, and whether gaming-related mental-health discussion appears as a distinct author community.

## Research Questions

Core GNN questions:

1. Can a multi-relational GNN improve mental-health classification compared with text-only models?
2. Do `college`, `gaming`, and `mental_health` edge types contribute differently to node classification performance?
3. Which relation type is most useful for predicting mental-health status?
4. Does combining all three relation types improve classification more than using any single relation type alone?
5. Can graph structure recover weak or hidden mental-health signals that are not obvious from text embeddings alone?

Author-node unlabeled questions:

1. What author communities emerge when users are connected by college, gaming, and mental-health relations?
2. Do authors who discuss both college and gaming occupy bridge positions near mental-health communities?
3. Are there distinct author groups around academic stress, gaming escapism, social isolation, or crisis language?
4. Can self-supervised GNN embeddings separate authors with occasional mental-health posts from authors with repeated mental-health distress?
5. Which relation type creates the strongest community structure among authors?

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
