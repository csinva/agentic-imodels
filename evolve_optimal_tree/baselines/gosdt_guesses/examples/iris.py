import gosdt
from sklearn.datasets import load_iris

# Load the dataset
X, y = load_iris(return_X_y=True, as_frame=True)
enc = gosdt.NumericBinarizer()
X_bin = enc.set_output(transform='pandas').fit_transform(X)

# Train the GOSDT classifier
clf = gosdt.GOSDTClassifier(
    regularization=0.1, depth_budget=5, verbose=True, worker_limit=20)
clf.fit(X_bin, y)
for tree in clf.trees_:
    print(tree)
