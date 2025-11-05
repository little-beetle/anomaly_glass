import numpy as np, glob

X_files = sorted(glob.glob("embeddings/features_batch_*.npy"))
y_files = sorted(glob.glob("embeddings/labels_batch_*.npy"))

X = np.concatenate([np.load(f) for f in X_files])
y = np.concatenate([np.load(f) for f in y_files])

np.save("embeddings/features_test.npy", X)
np.save("embeddings/labels_test.npy", y)

print("✅ Об'єднано:", X.shape, y.shape, "унікальні мітки:", np.unique(y))
