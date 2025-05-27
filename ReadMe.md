# ML Training Platform - Comprehensive Usage Guide

This guide provides complete examples for all endpoints and data types supported by the ML Training Platform API.

## 🚀 Quick Start

```bash
# Start the server
python main.py

# API will be available at: http://localhost:8000
# Interactive docs at: http://localhost:8000/docs
```

---

## 📊 Dataset Upload Examples

### 1. Tabular Classification Data

```bash
# Upload CSV for classification
curl -X POST "http://localhost:8000/upload/dataset" \
  -F "file=@iris_dataset.csv" \
  -F "data_type=tabular" \
  -F "task_type=classification" \
  -F "user_name=john_doe" \
  -F "target_column=species"
```

**Expected CSV format:**
```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
6.2,3.4,5.4,2.3,virginica
```

**Response:**
```json
{
  "dataset_id": "uuid-here",
  "data_type": "tabular",
  "task_type": "classification",
  "shape": [150, 5],
  "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
  "target_info": {
    "type": "categorical",
    "num_classes": 3,
    "classes": ["setosa", "versicolor", "virginica"],
    "distribution": {"setosa": 50, "versicolor": 50, "virginica": 50}
  }
}
```

### 2. Tabular Regression Data

```bash
# Upload CSV for regression
curl -X POST "http://localhost:8000/upload/dataset" \
  -F "file=@house_prices.csv" \
  -F "data_type=tabular" \
  -F "task_type=regression" \
  -F "user_name=jane_smith" \
  -F "target_column=price"
```

**Expected CSV format:**
```csv
bedrooms,bathrooms,sqft,age,price
3,2,1500,10,250000
4,3,2000,5,350000
2,1,1000,20,180000
```

### 3. Image Classification Data

```bash
# Upload ZIP file with class folders
curl -X POST "http://localhost:8000/upload/dataset" \
  -F "file=@image_classification.zip" \
  -F "data_type=image" \
  -F "task_type=classification" \
  -F "user_name=alex_vision"
```

**Expected ZIP structure:**
```
image_classification.zip
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── cat3.jpg
├── dogs/
│   ├── dog1.jpg
│   ├── dog2.jpg
│   └── dog3.jpg
└── birds/
    ├── bird1.jpg
    └── bird2.jpg
```

### 4. YOLO Object Detection Data

```bash
# Upload ZIP file for object detection
curl -X POST "http://localhost:8000/upload/dataset" \
  -F "file=@yolo_detection.zip" \
  -F "data_type=image" \
  -F "task_type=detection" \
  -F "user_name=detect_master"
```

---

## 🔍 Dataset Information & Management

### Get Dataset Information
```bash
curl -X GET "http://localhost:8000/dataset/info/{dataset_id}?user_name=john_doe"
```

### List User Datasets
```bash
curl -X GET "http://localhost:8000/datasets/john_doe"
```

**Response:**
```json
{
  "user_name": "john_doe",
  "datasets": [
    {
      "dataset_id": "uuid-1",
      "task_type": "classification",
      "uploaded_at": 1640995200
    },
    {
      "dataset_id": "uuid-2", 
      "task_type": "regression",
      "uploaded_at": 1640995300
    }
  ]
}
```

### Delete Dataset
```bash
curl -X DELETE "http://localhost:8000/datasets/{dataset_id}/john_doe"
```

---

## ⚙️ Data Preprocessing

### Basic Preprocessing
```bash
curl -X POST "http://localhost:8000/preprocess/{dataset_id}" \
  -H "Content-Type: application/json" \
  -d '{
    "user_name": "john_doe",
    "missing_values": "simple",
    "scaling": "standard",
    "encoding": "onehot"
  }'
```

### Advanced Preprocessing
```bash
curl -X POST "http://localhost:8000/preprocess/{dataset_id}" \
  -H "Content-Type: application/json" \
  -d '{
    "user_name": "data_scientist",
    "missing_values": "knn",
    "outliers": "isolation_forest",
    "scaling": "robust",
    "encoding": "onehot",
    "feature_selection": {
      "method": "f_classif",
      "k": 15
    },
    "dimensionality_reduction": {
      "method": "pca",
      "n_components": 0.95
    }
  }'
```

### Image Preprocessing
```bash
curl -X POST "http://localhost:8000/preprocess/{dataset_id}" \
  -H "Content-Type: application/json" \
  -d '{
    "user_name": "vision_user",
    "target_size": [224, 224],
    "augmentation": {
      "brightness": true,
      "contrast": true,
      "blur": false
    }
  }'
```

---

## 🤖 Model Training Examples

### 1. Random Forest Classification

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "RandomForest",
    "user_name": "john_doe",
    "dataset_id": "uuid-here",
    "model_name": "iris_classifier",
    "task_type": "classification",
    "parameters": {
      "n_estimators": 100,
      "max_depth": 10,
      "min_samples_split": 5,
      "random_state": 42
    }
  }'
```

### 2. Neural Network Regression

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "ANNRegressor",
    "user_name": "jane_smith",
    "dataset_id": "uuid-here",
    "model_name": "house_price_predictor",
    "task_type": "regression",
    "parameters": {
      "depth": 4,
      "hidden_size": 256,
      "epochs": 150,
      "batch_size": 32,
      "learning_rate": 0.001,
      "early_stopping_patience": 15
    }
  }'
```

### 3. CNN Image Classification

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "CNN",
    "user_name": "alex_vision",
    "dataset_id": "uuid-here",
    "model_name": "animal_classifier",
    "task_type": "classification",
    "parameters": {
      "depth": 5,
      "epochs": 50,
      "batch_size": 32,
      "learning_rate": 0.0001
    }
  }'
```

### 4. ResNet Transfer Learning

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "ResNet",
    "user_name": "transfer_expert",
    "dataset_id": "uuid-here",
    "model_name": "fine_tuned_resnet",
    "task_type": "classification",
    "parameters": {
      "variant": "resnet50",
      "freeze_backbone": true,
      "epochs": 25,
      "batch_size": 16,
      "learning_rate": 0.0001
    }
  }'
```

### 5. YOLO Object Detection

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "YOLO",
    "user_name": "detect_master",
    "dataset_id": "uuid-here",
    "model_name": "custom_detector",
    "task_type": "detection",
    "parameters": {
      "epochs": 100,
      "img_size": 640,
      "batch_size": 16
    }
  }'
```

### 6. Support Vector Regression

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "SVR",
    "user_name": "regression_pro",
    "dataset_id": "uuid-here",
    "model_name": "svr_model",
    "task_type": "regression",
    "parameters": {
      "kernel": "rbf",
      "C": 1.0,
      "epsilon": 0.1,
      "gamma": "scale"
    }
  }'
```

### 7. Gradient Boosting

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "GradientBoosting",
    "user_name": "ensemble_master",
    "dataset_id": "uuid-here",
    "model_name": "gb_regressor",
    "task_type": "regression",
    "parameters": {
      "n_estimators": 200,
      "learning_rate": 0.1,
      "max_depth": 6,
      "subsample": 0.8
    }
  }'
```

---

## 📈 Training Monitoring

### Get Training Status
```bash
curl -X GET "http://localhost:8000/training/status/{task_id}"
```

**Response:**
```json
{
  "task_id": "training-uuid",
  "status": "running",
  "progress": 65.0,
  "message": "Epoch 13/20, Train Acc: 0.8245, Val Acc: 0.7890",
  "metrics": {
    "loss": [0.8, 0.6, 0.4, 0.3],
    "accuracy": [0.6, 0.7, 0.8, 0.82],
    "val_loss": [0.9, 0.7, 0.5, 0.4],
    "val_accuracy": [0.55, 0.68, 0.75, 0.79]
  },
  "visualizations": [
    "training/task-id/training_loss.png",
    "training/task-id/accuracy_curves.png"
  ]
}
```

### Get Training Metrics
```bash
curl -X GET "http://localhost:8000/training/metrics/{task_id}"
```

### Get Training Visualizations
```bash
curl -X GET "http://localhost:8000/training/visualizations/{task_id}"
```

---

## 🎯 Model Prediction

### Tabular Data Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "iris_classifier",
    "user_name": "john_doe",
    "data": [
      [5.1, 3.5, 1.4, 0.2],
      [6.2, 3.4, 5.4, 2.3],
      [7.0, 3.2, 4.7, 1.4]
    ]
  }'
```

**Response:**
```json
{
  "model_name": "iris_classifier",
  "model_type": "RandomForest",
  "task_type": "classification",
  "predictions": [
    {
      "prediction": "setosa",
      "probabilities": [0.95, 0.03, 0.02]
    },
    {
      "prediction": "virginica", 
      "probabilities": [0.01, 0.15, 0.84]
    },
    {
      "prediction": "versicolor",
      "probabilities": [0.02, 0.88, 0.10]
    }
  ],
  "status": "success"
}
```

### Image Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "animal_classifier",
    "user_name": "alex_vision",
    "data": {
      "image_path": "/path/to/test_image.jpg"
    }
  }'
```

**Response:**
```json
{
  "model_name": "animal_classifier",
  "model_type": "CNN",
  "task_type": "classification",
  "predictions": {
    "predicted_class": "dog",
    "confidence": 0.8924,
    "all_probabilities": [0.0156, 0.8924, 0.0920]
  },
  "status": "success"
}
```

### Regression Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "house_price_predictor",
    "user_name": "jane_smith", 
    "data": [
      [3, 2, 1500, 10],
      [4, 3, 2000, 5]
    ]
  }'
```

**Response:**
```json
{
  "model_name": "house_price_predictor",
  "model_type": "ANNRegressor",
  "task_type": "regression",
  "predictions": [245000.75, 342150.25],
  "status": "success"
}
```

---

## 📊 Model Evaluation

### Evaluate with Test Data
```bash
curl -X POST "http://localhost:8000/evaluate/iris_classifier" \
  -F "user_name=john_doe" \
  -F "test_data=@test_set.csv"
```

**Response:**
```json
{
  "model_name": "iris_classifier",
  "evaluation_metrics": {
    "accuracy": 0.9333,
    "classification_report": {
      "setosa": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0},
      "versicolor": {"precision": 0.9, "recall": 0.9, "f1-score": 0.9},
      "virginica": {"precision": 0.9, "recall": 0.9, "f1-score": 0.9}
    }
  },
  "visualizations": [
    "evaluation_iris_classifier/confusion_matrix.png",
    "evaluation_iris_classifier/classification_report.png",
    "evaluation_iris_classifier/roc_curves_multiclass.png"
  ],
  "test_samples": 30,
  "status": "success"
}
```

---

## 🔧 Model Postprocessing

### Probability Calibration
```bash
curl -X POST "http://localhost:8000/postprocess/iris_classifier" \
  -H "Content-Type: application/json" \
  -d '{
    "user_name": "john_doe",
    "calibration": true,
    "calibration_method": "sigmoid",
    "calibration_cv": 3
  }'
```

### Threshold Optimization
```bash
curl -X POST "http://localhost:8000/postprocess/binary_classifier" \
  -H "Content-Type: application/json" \
  -d '{
    "user_name": "optimizer",
    "threshold_optimization": true,
    "threshold_metric": "f1"
  }'
```

### Model Pruning & Quantization
```bash
curl -X POST "http://localhost:8000/postprocess/cnn_model" \
  -H "Content-Type: application/json" \
  -d '{
    "user_name": "efficiency_expert",
    "model_pruning": true,
    "pruning_amount": 0.3,
    "pruning_method": "magnitude",
    "quantization": true,
    "quantization_type": "dynamic"
  }'
```

### Feature Importance Analysis
```bash
curl -X POST "http://localhost:8000/postprocess/rf_model" \
  -H "Content-Type: application/json" \
  -d '{
    "user_name": "analyst",
    "feature_importance": true
  }'
```

**Response:**
```json
{
  "model_name": "rf_model",
  "postprocessing_steps": ["feature_importance_analysis"],
  "improvements": {
    "feature_importance": {
      "method": "built_in_importance",
      "top_features": [
        ["petal_length", 0.45],
        ["petal_width", 0.32],
        ["sepal_length", 0.15],
        ["sepal_width", 0.08]
      ],
      "total_features": 4,
      "top_10_contribution": 1.0
    }
  },
  "status": "completed"
}
```

---

## 📈 Visualization Generation

### Dataset Visualizations
```bash
curl -X GET "http://localhost:8000/visualizations/dataset/{dataset_id}?user_name=john_doe"
```

**Response:**
```json
{
  "visualizations": [
    "john_doe/dataset_uuid/data_distribution.png",
    "john_doe/dataset_uuid/target_distribution.png", 
    "john_doe/dataset_uuid/correlation_matrix.png",
    "john_doe/dataset_uuid/feature_statistics.png"
  ]
}
```

---

## 🗂️ Model Management

### List User Models
```bash
curl -X GET "http://localhost:8000/models/john_doe"
```

**Response:**
```json
{
  "user_name": "john_doe",
  "models": [
    {
      "model_name": "iris_classifier",
      "model_type": "RandomForest", 
      "task_type": "classification",
      "metrics": {"accuracy": 0.9333},
      "file_size": 2048576,
      "created_at": 1640995200
    },
    {
      "model_name": "price_predictor",
      "model_type": "ANNRegressor",
      "task_type": "regression", 
      "metrics": {"r2": 0.8456, "mse": 125000},
      "file_size": 1024000,
      "created_at": 1640995300
    }
  ]
}
```

### Get Model Information
```bash
curl -X GET "http://localhost:8000/model/info/iris_classifier/john_doe"
```

### Download Model
```bash
curl -X GET "http://localhost:8000/download/model/iris_classifier/john_doe" \
  --output iris_classifier_john_doe.zip
```

### Delete Model
```bash
curl -X DELETE "http://localhost:8000/models/iris_classifier/john_doe"
```

---

## 🔍 System Information

### Get Supported Models
```bash
curl -X GET "http://localhost:8000/supported_models"
```

**Response:**
```json
{
  "classification": {
    "YOLO": {
      "description": "Object detection and classification",
      "parameters": ["epochs", "img_size", "batch_size"]
    },
    "CNN": {
      "description": "Convolutional Neural Network",
      "parameters": ["depth", "epochs", "batch_size", "learning_rate"]
    },
    "ResNet": {
      "description": "Residual Network", 
      "parameters": ["variant", "epochs", "batch_size", "learning_rate"]
    },
    "RandomForest": {
      "description": "Random Forest Classifier",
      "parameters": ["n_estimators", "max_depth", "min_samples_split"]
    },
    "ANN": {
      "description": "Artificial Neural Network",
      "parameters": ["depth", "hidden_size", "epochs", "batch_size"]
    }
  },
  "regression": {
    "LinearRegression": {
      "description": "Linear Regression",
      "parameters": ["fit_intercept", "normalize"]
    },
    "RandomForestRegressor": {
      "description": "Random Forest Regressor", 
      "parameters": ["n_estimators", "max_depth", "min_samples_split"]
    },
    "ANNRegressor": {
      "description": "Neural Network Regressor",
      "parameters": ["depth", "hidden_size", "epochs", "batch_size"]
    },
    "SVR": {
      "description": "Support Vector Regression",
      "parameters": ["kernel", "C", "epsilon", "gamma"]
    },
    "GradientBoosting": {
      "description": "Gradient Boosting Regressor",
      "parameters": ["n_estimators", "learning_rate", "max_depth"]
    }
  }
}
```

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### API Root
```bash
curl -X GET "http://localhost:8000/"
```

---

## 🐍 Python Client Examples

### Using requests library

```python
import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"
USER_NAME = "python_user"

# Upload dataset
with open('data.csv', 'rb') as f:
    files = {'file': f}
    data = {
        'data_type': 'tabular',
        'task_type': 'classification', 
        'user_name': USER_NAME,
        'target_column': 'label'
    }
    response = requests.post(f"{BASE_URL}/upload/dataset", files=files, data=data)
    dataset_info = response.json()
    dataset_id = dataset_info['dataset_id']

# Train model
train_config = {
    "model_type": "RandomForest",
    "user_name": USER_NAME,
    "dataset_id": dataset_id,
    "model_name": "my_classifier",
    "task_type": "classification",
    "parameters": {
        "n_estimators": 100,
        "max_depth": 10
    }
}

response = requests.post(f"{BASE_URL}/train", json=train_config)
task_info = response.json()
task_id = task_info['task_id']

# Monitor training
import time
while True:
    response = requests.get(f"{BASE_URL}/training/status/{task_id}")
    status = response.json()
    print(f"Progress: {status['progress']:.1f}% - {status['message']}")
    
    if status['status'] in ['completed', 'failed']:
        break
    time.sleep(5)

# Make predictions
prediction_data = {
    "model_name": "my_classifier",
    "user_name": USER_NAME,
    "data": [[5.1, 3.5, 1.4, 0.2]]
}

response = requests.post(f"{BASE_URL}/predict", json=prediction_data)
predictions = response.json()
print("Predictions:", predictions)
```

---

## 📋 Complete Workflow Examples

### Image Classification Workflow

```bash
# 1. Upload image dataset
curl -X POST "http://localhost:8000/upload/dataset" \
  -F "file=@animals.zip" \
  -F "data_type=image" \
  -F "task_type=classification" \
  -F "user_name=vision_expert"

# 2. Preprocess images  
curl -X POST "http://localhost:8000/preprocess/{dataset_id}" \
  -H "Content-Type: application/json" \
  -d '{
    "user_name": "vision_expert",
    "target_size": [224, 224],
    "augmentation": {"brightness": true, "contrast": true}
  }'

# 3. Train CNN model
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "CNN",
    "user_name": "vision_expert", 
    "dataset_id": "{dataset_id}",
    "model_name": "animal_cnn",
    "task_type": "classification",
    "parameters": {"depth": 4, "epochs": 30, "batch_size": 32}
  }'

# 4. Monitor training progress
curl -X GET "http://localhost:8000/training/status/{task_id}"

# 5. Evaluate model
curl -X POST "http://localhost:8000/evaluate/animal_cnn" \
  -F "user_name=vision_expert" \
  -F "test_data=@test_images.csv"

# 6. Make predictions
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "animal_cnn",
    "user_name": "vision_expert",
    "data": {"image_path": "/path/to/test.jpg"}
  }'
```

### Regression Analysis Workflow

```bash
# 1. Upload regression dataset
curl -X POST "http://localhost:8000/upload/dataset" \
  -F "file=@housing.csv" \
  -F "data_type=tabular" \
  -F "task_type=regression" \
  -F "user_name=data_analyst" \
  -F "target_column=price"

# 2. Advanced preprocessing
curl -X POST "http://localhost:8000/preprocess/{dataset_id}" \
  -H "Content-Type: application/json" \
  -d '{
    "user_name": "data_analyst",
    "missing_values": "knn",
    "outliers": "iqr", 
    "scaling": "robust",
    "feature_selection": {"method": "f_regression", "k": 15}
  }'

# 3. Train multiple models and compare
# Neural Network
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "ANNRegressor",
    "user_name": "data_analyst",
    "dataset_id": "{dataset_id}",
    "model_name": "nn_regressor",
    "task_type": "regression",
    "parameters": {"depth": 5, "hidden_size": 256, "epochs": 100}
  }'

# Random Forest
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "RandomForestRegressor", 
    "user_name": "data_analyst",
    "dataset_id": "{dataset_id}",
    "model_name": "rf_regressor",
    "task_type": "regression",
    "parameters": {"n_estimators": 200, "max_depth": 15}
  }'

# 4. Postprocess best model
curl -X POST "http://localhost:8000/postprocess/nn_regressor" \
  -H "Content-Type: application/json" \
  -d '{
    "user_name": "data_analyst",
    "feature_importance": true,
    "prediction_intervals": true,
    "confidence_level": 0.95
  }'

# 5. Generate comprehensive visualizations
curl -X GET "http://localhost:8000/visualizations/dataset/{dataset_id}?user_name=data_analyst"
```

---

## ⚠️ Error Handling

### Common Error Responses

**Dataset not found:**
```json
{
  "detail": "Dataset not found",
  "status_code": 404
}
```

**Invalid file format:**
```json
{
  "detail": "Unsupported file format for tabular data", 
  "status_code": 400
}
```

**Training failed:**
```json
{
  "detail": "Training failed: Insufficient data for training",
  "status_code": 400
}
```

**Model not found:**
```json
{
  "detail": "Model not found",
  "status_code": 404
}
```

---

## 🚀 Best Practices

### 1. Dataset Organization
- **Tabular**: Include target column, handle missing values
- **Images**: Organize in class folders, consistent naming
- **File sizes**: Keep under 500MB per upload

### 2. Training Tips
- Start with smaller models for quick testing
- Use validation data for hyperparameter tuning
- Monitor training progress regularly
- Save intermediate results

### 3. Production Deployment
- Use proper authentication for user management
- Implement rate limiting for API endpoints
- Set up model versioning and rollback
- Monitor resource usage and scaling

### 4. Performance Optimization
- Use preprocessing pipelines for consistency
- Apply postprocessing for better model performance
- Consider ensemble methods for critical applications
- Implement caching for frequently used models

---

This comprehensive guide covers all the endpoints and functionality available in the ML Training Platform. Each example includes realistic data formats, parameters, and expected responses to help you get started quickly with any type of machine learning task.