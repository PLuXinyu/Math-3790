# This code facenet model is sourced from the GitHub repository by Tim Esler(2018).
# URL: https://github.com/timesler/facenet-pytorch

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import zipfile
import cv2

# Initialize face detector and embedding model
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load images from ZIP file
faces = {}
with zipfile.ZipFile("attface.zip") as facezip:
    for filename in facezip.namelist():
        if not filename.endswith(".pgm"):
            continue
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

# Prepare training and testing data
train_facematrix = []
train_facelabel = []
test_faces = {}
test_labels = []

for key, val in faces.items():
    if key.endswith("10.pgm") or key.endswith("9.pgm"):
        test_faces[key] = val
        test_labels.append(key.split("/")[0])
    else:
        train_facematrix.append(val)
        train_facelabel.append(key.split("/")[0])

# Generate embedding vectors
def get_embedding(mtcnn, resnet, img):
    # Convert PIL image to RGB format if not already
    img = img.convert('RGB')
    # Use MTCNN to detect and crop face
    img_cropped = mtcnn(img)
    if img_cropped is None:
        return None
    # Check and convert image to 4D tensor
    if img_cropped.ndim == 3:
        img_cropped = img_cropped.unsqueeze(0)
    # Compute embedding vector
    img_embedding = resnet(img_cropped)
    return img_embedding.detach().numpy()[0]

# Process training data
train_embeddings = []
for img in train_facematrix:
    pil_img = Image.fromarray(img)
    embedding = get_embedding(mtcnn, resnet, pil_img)
    if embedding is not None:
        train_embeddings.append(embedding)
train_embeddings = np.array(train_embeddings)

# Process testing data
test_embeddings = []
for key, img in test_faces.items():
    pil_img = Image.fromarray(img)
    embedding = get_embedding(mtcnn, resnet, pil_img)
    if embedding is not None:
        test_embeddings.append(embedding)
test_embeddings = np.array(test_embeddings)

# Convert lists to numpy arrays
train_embeddings = np.array(train_embeddings)
test_embeddings = np.array(test_embeddings)

# Standardize data
scaler = StandardScaler()
train_embeddings = scaler.fit_transform(train_embeddings)
test_embeddings = scaler.transform(test_embeddings)

# Use GridSearchCV to find the best value of k
param_grid = {'n_neighbors': np.arange(1, 11)}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(train_embeddings, train_facelabel)
best_k = grid_search.best_params_['n_neighbors']
print(f"Best value of K: {best_k}")

# Train k-NN classifier with the best k value
best_knn = grid_search.best_estimator_

# Test a query image and return if the prediction is correct
def test_query_image(query_image_key):
    pil_img = Image.fromarray(test_faces[query_image_key])
    query_embedding = get_embedding(mtcnn, resnet, pil_img)
    if query_embedding is None:
        return False
    query_embedding = scaler.transform([query_embedding])
    prediction = best_knn.predict(query_embedding)[0]
    return prediction == query_image_key.split("/")[0]

# Calculate accuracy
correct_predictions = 0
for key in test_faces.keys():
    if test_query_image(key):
        correct_predictions += 1

accuracy = correct_predictions / len(test_faces)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize results
plt.figure(figsize=(10, 8))
for i in range(min(50, len(test_faces))):
    ax = plt.subplot(10, 5, i + 1)
    key = list(test_faces.keys())[i]
    ax.imshow(test_faces[key], cmap="gray")
    ax.axis("off")
    true_label = key.split("/")[0]
    pil_img = Image.fromarray(test_faces[key])
    query_embedding = get_embedding(mtcnn, resnet, pil_img)
    if query_embedding is not None:
        query_embedding = scaler.transform([query_embedding])
        predict_label = best_knn.predict(query_embedding)[0]
        plt.title(f"True: {true_label}\nPredict: {predict_label}")
plt.show()

# Display a random face embedding as an image
embedding_image = train_embeddings[0].reshape((32, 16))  # Reshape to 32x16 for visualization
plt.imshow(embedding_image, cmap="gray")
plt.title("Feature Image from Embedding")
plt.axis("off")
plt.show()

# Plot k-NN parameters vs. accuracy
results = grid_search.cv_results_
scores = results['mean_test_score']

plt.figure(figsize=(10, 6))
plt.plot(param_grid['n_neighbors'], scores, marker='o')
plt.title('Accuracy vs. Number of Neighbors (K)')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
