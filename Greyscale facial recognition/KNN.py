import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Load images from zip file
faces = {}
with zipfile.ZipFile("attface.zip") as facezip:
    for filename in facezip.namelist():
        if not filename.endswith(".pgm"):
            continue  # not a face picture
        with facezip.open(filename) as image:
            faces[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

# Display some of the images
total_images = len(faces)
start_index = total_images // 2 - 8
faceimages = list(faces.values())[start_index:start_index + 16]

fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
for i in range(16):
    axes[i % 4][i // 4].imshow(faceimages[i], cmap="gray")
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
    if key.endswith("10.pgm") or key.endswith("9.pgm"):
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

# Show the first 16 eigenfaces
fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
for i in range(16):
    axes[i % 4][i // 4].imshow(pca.components_[i].reshape(faceshape), cmap="gray")
plt.show()

# Find the best value of K using GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 11)}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(train_facematrix_pca, train_facelabel)
best_k = grid_search.best_params_['n_neighbors']
print(f"Best value of K: {best_k}")

# Train the best KNN model
best_knn = grid_search.best_estimator_
best_knn.fit(train_facematrix_pca, train_facelabel)

# Test on a query image and return if the prediction is correct
def test_query_image(query_image_key):
    query_image = test_faces[query_image_key].flatten().reshape(1, -1)
    query_image = scaler.transform(query_image)
    query_image_pca = pca.transform(query_image)
    prediction = best_knn.predict(query_image_pca)[0]
    return prediction == query_image_key.split("/")[0]

# Calculate accuracy
correct_predictions = 0
for key in test_faces.keys():
    if test_query_image(key):
        correct_predictions += 1

accuracy = correct_predictions / len(test_faces)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plot accuracy vs K values
k_values = np.arange(1, 11)
accuracies = grid_search.cv_results_['mean_test_score']

# Ensure k_values and accuracies have the same length
k_values = k_values[:len(accuracies)]

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('Accuracy vs. Number of Neighbors (K)')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
