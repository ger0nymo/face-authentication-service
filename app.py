import io
import json
import torch.nn.functional
import numpy as np

from flask import Flask, jsonify, request, abort
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

app = Flask(__name__)

mtcnn = MTCNN()

resnet = InceptionResnetV1(pretrained='casia-webface').eval()

API_KEY = 'super-secret-api-key'  # Obviously in a real-world scenario, this should be stored in a secure way


def authenticate_request(req):
    api_key = req.headers.get('key')

    print(f"Authenticated request with API key: {api_key}")

    if api_key != API_KEY:
        abort(401, 'Unauthorized')


@app.route('/image-embedding', methods=['POST'])
def image_embedding():
    authenticate_request(request)

    print(f"Request: {request.files}")

    # Get the uploaded image from the request, check if it contains one face only, and create embedding
    uploaded_image = request.files['image'].read()
    uploaded_image = Image.open(io.BytesIO(uploaded_image))

    # Detect faces in the uploaded image
    boxes, _ = mtcnn.detect(uploaded_image)

    if boxes is None:
        return jsonify({'error': 'No face detected in the uploaded image'})

    # If no faces or too many faces are detected, return an error message
    if len(boxes) == 0:
        return jsonify({'error': 'No face detected in the uploaded image'})

    if len(boxes) > 1:
        return jsonify({'error': 'Multiple faces detected in the uploaded image'})

    # Align and get the embedding for the uploaded image
    aligned_image = mtcnn(uploaded_image)
    uploaded_embedding = resnet(aligned_image.unsqueeze(0)).detach()

    print(f"Shape: {uploaded_embedding.shape}")

    # Return the embedding as the response
    return jsonify({'fv': uploaded_embedding.flatten().tolist()})


@app.route('/compare-faces', methods=['POST'])
def compare_faces():
    authenticate_request(request)

    print(f"Request: {request.form['fv']}")

    # Get the registered embeddingA from the request form
    registered_embedding = json.loads(request.form['fv'])
    registered_embedding = torch.tensor(registered_embedding)
    print(f"Registered embedding: {registered_embedding.shape}")

    # Get the uploaded image from the request
    uploaded_image = request.files['image'].read()
    uploaded_image = Image.open(io.BytesIO(uploaded_image))

    # Detect faces in the uploaded image
    boxes, _ = mtcnn.detect(uploaded_image)

    if boxes is None:
        return jsonify({'error': 'No faces detected in the uploaded image'})

    print(f"Found {len(boxes)} face(s)")

    # If no faces or too many faces are detected, return an error message
    if len(boxes) == 0:
        return jsonify({'error': 'No faces detected in the uploaded image'})

    if len(boxes) > 1:
        return jsonify({'error': 'Multiple faces detected in the uploaded image'})

    # Align and get the embedding for the uploaded image
    aligned_image = mtcnn(uploaded_image)
    uploaded_embedding = resnet(aligned_image.unsqueeze(0)).detach()

    # Calculate the cosine similarity between the registered vector and uploaded image's embedding
    cosine_sim = np.dot(registered_embedding, uploaded_embedding.T).flatten()

    # Return the cosine similarity as the response
    return jsonify({'cosine_similarity': cosine_sim.item()})


if __name__ == '__main__':
    app.run(port=5000)
