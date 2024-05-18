# Flask API for a face recognition based authentication system

## Introduction
I created this as part of a college project laboratory project.
The project is a face recognition based authentication system.

This API is capable creating embedding vectors of faces and also comparing an image with an embedding vector. 

The app utilizes Flask and the [facenet-pytorch](https://github.com/timesler/facenet-pytorch) library to create rest endpoints that can create the embedding vectors and do the comparisons.

## Installation

1. Clone the repository
2. Install the requirements
    ```bash
    pip install -r requirements.txt
    ```
3. Run the app
    ```bash
    flask --debug run
    ```

## Example requests
Every request should contain an api key in the headers. 

The API key is known only to the authentication backend API and this API.

### Create an embedding vector
```http request
POST /image-embedding
```

Payload type: form-data

```typescript
interface Payload {
    image: File,
}
```

Expected successful result: Status code - 200

```json
{
  "fv": [...] 
}
```
It should return a 512 dimensional embedding vector.

### Compare an image with an embedding vector
```http request
POST /compare-faces
```

Payload type: form-data

```typescript
interface Payload {
    image: File,
    fv: string, // The embedding vector as a string in { "fv": [...] } format
}
```

Expected successful result: Status code - 200

```json
{
  "cosine_similarity": 0.9
}
```
It should return the cosine similarity between the image and the embedding vector.