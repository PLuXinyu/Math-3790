import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Load images from zip file
faces = {}
with zipfile.ZipFile("attface.zip") as facezip:
    for filename in facezip.namelist():
        if not filename.endswith(".pgm"):
            continue  # not a face picture
        with facezip.open(filename) as image:
            faces[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

# Display all images
fig, axes = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(15, 15))
faceimages = list(faces.values())
for i, ax in enumerate(axes.flat):
    if i < len(faceimages):
        ax.imshow(faceimages[i], cmap="gray")
        ax.axis('off')
plt.show()

# Print some information about the dataset
faceshape = list(faces.values())[0].shape
print("Face image shape:", faceshape)
print(list(faces.keys())[:5])
classes = set(filename.split("/")[0] for filename in faces.keys())
print("Number of classes:", len(classes))
print("Number of pictures:", len(faces))

# Prepare data for PCA
train_facematrix = []
train_facelabel = []
test_faces = {}
test_labels = []

for key, val in faces.items():
    if key.endswith("1.pgm"):
        test_faces[key] = val
        test_labels.append(key.split("/")[0])
    else:
        train_facematrix.append(val.flatten())
        train_facelabel.append(key.split("/")[0])

# Standardize data
scaler = StandardScaler()
train_facematrix = scaler.fit_transform(train_facematrix)

# Apply PCA to extract eigenfaces
pca = PCA(n_components=50).fit(train_facematrix)
train_facematrix_pca = pca.transform(train_facematrix)
print(pca.explained_variance_ratio_)

# Show a new set of 16 eigenfaces
fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
for i in range(16):
    axes[i % 4][i // 4].imshow(pca.components_[(i + 16) % pca.components_.shape[0]].reshape(faceshape), cmap="gray")
plt.show()

# Train SVM classifier with GridSearchCV
svc = SVC()
params = {"C": np.logspace(-3, 3, 10), "kernel": ["rbf", "linear"]}
gc = GridSearchCV(estimator=svc, param_grid=params, cv=5)
gc.fit(train_facematrix_pca, train_facelabel)
print("Best parameters:", gc.best_params_)

# Get best estimator from GridSearchCV
best_svc = gc.best_estimator_

# Test on a query image and return if the prediction is correct
def test_query_image(query_image_key):
    query_image = test_faces[query_image_key].flatten().reshape(1, -1)
    query_image = scaler.transform(query_image)
    query_image_pca = pca.transform(query_image)
    prediction = best_svc.predict(query_image_pca)[0]
    return prediction == query_image_key.split("/")[0]

# Calculate accuracy
correct_predictions = 0
for key in test_faces.keys():
    if test_query_image(key):
        correct_predictions += 1

accuracy = correct_predictions / len(test_faces)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize results with a new set of test images
plt.figure(figsize=(10, 8))
for i in range(min(50, len(test_faces))):
    ax = plt.subplot(10, 5, i + 1)
    key = list(test_faces.keys())[i]
    ax.imshow(test_faces[key], cmap="gray")
    ax.axis("off")
    true_label = key.split("/")[0]
    query_image = test_faces[key].flatten().reshape(1, -1)
    query_image = scaler.transform(query_image)
    query_image_pca = pca.transform(query_image)
    predict_label = best_svc.predict(query_image_pca)[0]
    plt.title(f"True: {true_label}\nPredict: {predict_label}")

plt.show()

