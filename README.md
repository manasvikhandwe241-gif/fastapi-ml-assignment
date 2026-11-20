

# Iris Classification REST API using FastAPI

## 1. Aim

To design, develop, test, and document a REST API using FastAPI for serving predictions from a trained machine learning model.

## 2. Problem Statement

A machine learning model developed in a Jupyter Notebook cannot be directly consumed by web applications or other software systems. This project exposes a trained Iris flower classification model through a REST API using FastAPI.

The API accepts flower measurements in JSON format and returns the predicted Iris species along with a confidence score and class probabilities.

## 3. Technologies Used

- Python 3.12
- FastAPI
- Scikit-learn
- NumPy
- Joblib
- Pydantic
- Pytest
- HTTPX
- Uvicorn

## 4. Project Structure

```text
fastapi-ml-assignment/
│
├── artifacts/
│   └── iris_model.joblib
│
├── tests/
│   └── test_api.py
│
├── main.py
├── train_model.py
├── requirements.txt
└── README.md
````

## 5. Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 6. Train the Machine Learning Model

Run:

```bash
python train_model.py
```

The trained model is saved as:

```text
artifacts/iris_model.joblib
```

The model used is a Random Forest Classifier.

The test accuracy obtained during training was:

```text
90%
```

## 7. Run the FastAPI Application

Start the FastAPI development server:

```bash
fastapi dev main.py
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## 8. API Endpoints

| Method | Endpoint      | Description                      |
| ------ | ------------- | -------------------------------- |
| GET    | `/`           | Displays basic API information   |
| GET    | `/health`     | Checks API and model health      |
| GET    | `/model-info` | Displays model metadata          |
| POST   | `/predict`    | Predicts the Iris flower species |

## 9. Swagger Documentation

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

ReDoc documentation is available at:

```text
http://127.0.0.1:8000/redoc
```

OpenAPI schema is available at:

```text
http://127.0.0.1:8000/openapi.json
```

## 10. Prediction Request

The `/predict` endpoint accepts JSON input.

Example:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

## 11. Prediction Response

Example response:

```json
{
  "predicted_class": 0,
  "predicted_species": "setosa",
  "confidence": 1.0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "model_version": "1.0.0"
}
```

## 12. Input Validation

The API validates all input values using Pydantic.

Each feature must:

* Be a numeric value.
* Be greater than 0.
* Be less than or equal to 20.

Invalid requests return HTTP status code `422`.

Examples of invalid input include:

* Negative feature values.
* Missing features.
* Text values instead of numbers.

## 13. Error Handling

The API uses appropriate HTTP status codes:

| Status Code | Meaning                        |
| ----------- | ------------------------------ |
| 200         | Request successfully processed |
| 404         | Endpoint not found             |
| 422         | Input validation failed        |
| 500         | Internal prediction failure    |

## 14. Automated Testing

The API was tested using Pytest.

Run:

```bash
python -m pytest -v
```

The automated test suite contains 8 test cases covering:

1. Root endpoint
2. Health endpoint
3. Model information endpoint
4. Valid prediction
5. Negative feature value
6. Missing feature
7. Incorrect data type
8. Unknown endpoint

All tests passed successfully.

```text
8 passed
```

## 15. Result

The Iris classification machine learning model was successfully trained, saved, and integrated with a FastAPI REST API.

The API successfully:

* Loads the trained model during application startup.
* Accepts JSON input.
* Validates input using Pydantic.
* Generates Iris species predictions.
* Returns confidence and probability values.
* Provides automatic Swagger documentation.
* Handles invalid requests.
* Passes automated API tests.

## 16. Future Deployment

The FastAPI application can be containerized using Docker and deployed to a cloud platform such as AWS for access by external applications.

```

Save it with **Ctrl + S**.

```
