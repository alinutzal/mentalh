import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

df = pd.read_csv("results/reddit_classified.csv")

y_true = df["status"]
y_pred = df["mental_health"]

# print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

# print("\nClassification Report:")
# print(
#     classification_report(
#         y_true,
#         y_pred,
#         labels=["Normal", "Depression", "Anxiety", "Suicidal"],
#         digits=4,
#     )
# )

# print("\nConfusion Matrix:")
# print(
#     confusion_matrix(
#         y_true,
#         y_pred,
#         labels=["Unclear", "Normal", "Depression", "Anxiety", "Suicidal"],
#     )
# )

# Count college posts
college_count = (df["college"] == True).sum()

# Count gaming posts
gaming_count = (df["gaming"] == True).sum()

# Count either college or gaming
either_count = ((df["college"] == True) | (df["gaming"] == True)).sum()

# Count both college and gaming
both_count = ((df["college"] == True) & (df["gaming"] == True)).sum()

print("\nTopic Counts")
print(f"College: {college_count:,}")
print(f"Gaming: {gaming_count:,}")
print(f"College OR Gaming: {either_count:,}")
print(f"College AND Gaming: {both_count:,}")

print("\nCollege posts:")
print(df[df["college"]]["mental_health"].value_counts())

print("\nGaming posts:")
print(df[df["gaming"]]["mental_health"].value_counts())

# Convert string booleans if needed
df["college"] = df["college"].astype(str).str.lower() == "true"
df["gaming"] = df["gaming"].astype(str).str.lower() == "true"

both = df[(df["college"]) & (df["gaming"])]

print(f"Posts with college=True and gaming=True: {len(both)}")

print("\nMental Health Distribution:")
print(both["mental_health"].value_counts())
