from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import os
import uuid
from models.data_handler import DataHandler
from models.model_trainer import ModelTrainer
from models.model_manager import ModelManager
from models.visualization import VisualizationManager
from models.preprocessing import PreprocessingManager
from models.postprocessing import PostprocessingManager
from utils.config import Config
from utils.logger import setup_logger

app = FastAPI(title="ML Training Platform", version="2.0.0")

Config.setup_directories()
logger = setup_logger()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

data_handler = DataHandler()
model_trainer = ModelTrainer()
model_manager = ModelManager()
viz_manager = VisualizationManager()
preprocessing_manager = PreprocessingManager()
postprocessing_manager = PostprocessingManager()

class DatasetInfo(BaseModel):
    dataset_id: str
    data_type: str
    task_type: str
    shape: tuple
    columns: Optional[List[str]] = None
    target_info: Dict[str, Any] = {}
    statistics: Dict[str, Any] = {}

class ModelConfig(BaseModel):
    model_type: str
    user_name: str
    dataset_id: str
    model_name: str
    task_type: str
    parameters: Dict[str, Any] = {}

class TrainingStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    metrics: Optional[Dict[str, Any]] = None
    visualizations: Optional[List[str]] = None

class PredictionRequest(BaseModel):
    model_name: str
    user_name: str
    data: Union[List[List[float]], Dict[str, Any]]

training_tasks = {}

@app.post("/upload/dataset", response_model=DatasetInfo)
async def upload_dataset(
    file: UploadFile = File(...),
    data_type: str = Form(...),
    task_type: str = Form(...),
    user_name: str = Form(...),
    target_column: Optional[str] = Form(None)
):
    try:
        dataset_info = await data_handler.upload_dataset(
            file, data_type, task_type, user_name, target_column
        )
        logger.info(f"Dataset uploaded: {dataset_info.dataset_id}")
        return dataset_info
    except Exception as e:
        logger.error(f"Dataset upload failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/dataset/info/{dataset_id}")
async def get_dataset_info(dataset_id: str, user_name: str):
    try:
        info = await data_handler.get_dataset_info(dataset_id, user_name)
        return info
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/preprocess/{dataset_id}")
async def preprocess_data(
    dataset_id: str,
    user_name: str,
    config: Dict[str, Any] = {}
):
    try:
        result = await preprocessing_manager.preprocess_dataset(
            dataset_id, user_name, config
        )
        logger.info(f"Preprocessing completed for dataset: {dataset_id}")
        return result
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train", response_model=TrainingStatus)
async def start_training(
    config: ModelConfig,
    background_tasks: BackgroundTasks
):
    task_id = str(uuid.uuid4())
    training_tasks[task_id] = {
        "status": "started",
        "progress": 0.0,
        "message": "Training initiated",
        "metrics": {},
        "visualizations": []
    }
    
    background_tasks.add_task(train_model_task, task_id, config)
    logger.info(f"Training started: {task_id}")
    
    return TrainingStatus(
        task_id=task_id,
        status="started",
        progress=0.0,
        message="Training initiated"
    )

async def train_model_task(task_id: str, config: ModelConfig):
    try:
        await model_trainer.train_model(task_id, config, training_tasks, viz_manager)
        logger.info(f"Training completed: {task_id}")
    except Exception as e:
        training_tasks[task_id] = {
            "status": "failed",
            "progress": 0.0,
            "message": str(e),
            "metrics": {},
            "visualizations": []
        }
        logger.error(f"Training failed: {task_id}, Error: {str(e)}")

@app.get("/training/status/{task_id}", response_model=TrainingStatus)
async def get_training_status(task_id: str):
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Training task not found")
    
    task_info = training_tasks[task_id]
    return TrainingStatus(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info["progress"],
        message=task_info["message"],
        metrics=task_info.get("metrics"),
        visualizations=task_info.get("visualizations", [])
    )

@app.get("/training/metrics/{task_id}")
async def get_training_metrics(task_id: str):
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Training task not found")
    
    return training_tasks[task_id].get("metrics", {})

@app.get("/training/visualizations/{task_id}")
async def get_training_visualizations(task_id: str):
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Training task not found")
    
    return training_tasks[task_id].get("visualizations", [])

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        result = await model_manager.predict(
            request.model_name, 
            request.user_name, 
            request.data
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/evaluate/{model_name}")
async def evaluate_model(
    model_name: str,
    user_name: str,
    test_data: UploadFile = File(...)
):
    try:
        evaluation_results = await model_manager.evaluate_model(
            model_name, user_name, test_data, viz_manager
        )
        return evaluation_results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/postprocess/{model_name}")
async def postprocess_model(
    model_name: str,
    user_name: str,
    config: Dict[str, Any] = {}
):
    try:
        result = await postprocessing_manager.postprocess_model(
            model_name, user_name, config
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models/{user_name}")
async def list_user_models(user_name: str):
    try:
        models = await model_manager.list_user_models(user_name)
        return models
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/info/{model_name}/{user_name}")
async def get_model_info(model_name: str, user_name: str):
    try:
        info = await model_manager.get_model_info(model_name, user_name)
        return info
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/download/model/{model_name}/{user_name}")
async def download_model(model_name: str, user_name: str):
    try:
        file_path = await model_manager.get_model_path(model_name, user_name)
        return FileResponse(
            path=file_path,
            filename=f"{model_name}_{user_name}.zip",
            media_type='application/zip'
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/models/{model_name}/{user_name}")
async def delete_model(model_name: str, user_name: str):
    try:
        await model_manager.delete_model(model_name, user_name)
        return {"message": f"Model {model_name} for user {user_name} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/datasets/{user_name}")
async def list_user_datasets(user_name: str):
    try:
        datasets = await data_handler.list_user_datasets(user_name)
        return datasets
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/datasets/{dataset_id}/{user_name}")
async def delete_dataset(dataset_id: str, user_name: str):
    try:
        await data_handler.delete_dataset(dataset_id, user_name)
        return {"message": f"Dataset {dataset_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/visualizations/dataset/{dataset_id}")
async def generate_dataset_visualizations(dataset_id: str, user_name: str):
    try:
        viz_paths = await viz_manager.generate_dataset_visualizations(
            dataset_id, user_name
        )
        return {"visualizations": viz_paths}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/supported_models")
async def get_supported_models():
    return {
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

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "directories": {
            "uploads": os.path.exists(Config.UPLOAD_DIR),
            "models": os.path.exists(Config.MODELS_DIR),
            "visualizations": os.path.exists(Config.VIZ_DIR)
        }
    }

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=8000, reload=True)