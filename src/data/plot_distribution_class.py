# pie_emotions.py — pie chart of emotion classes and amounts
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

# ---- Option A: load from CSV (FER-2013-style) ----
CSV_PATH = "fer2013.csv"  # <-- change to your path
df = pd.read_csv(CSV_PATH)           # expects a column named 'emotion' with values 0..6
counts = df["emotion"].value_counts().sort_index()

# Map numeric labels to names (FER-2013)
id_to_label = OrderedDict({
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral",
})
labels = [id_to_label.get(i, f"Class {i}") for i in counts.index]
sizes = counts.values

# ---- Pie chart (single plot; no explicit colors) ----
plt.figure(figsize=(6, 6))
plt.pie(
    sizes,
    labels=labels,
    autopct=lambda pct: f"{pct:.1f}%",
    startangle=90
)
plt.title("Emotion Class Distribution")
plt.tight_layout()
plt.savefig("emotion_class_distribution.png", dpi=300)
plt.show()

# count number of pictures per class
def count_class():
    for idx, label in id_to_label.items():
        count = counts.get(idx, 0)
        print(f"{label}: {count}")

    print(counts.sum())

count_class()