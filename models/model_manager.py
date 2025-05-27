import torch
import torch.nn as nn
import pandas as pd
import joblib
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Union
from fastapi import UploadFile, HTTPException
from PIL import Image
import torchvision.transforms as transforms
from utils.config import Config
from utils.logger import setup_logger
from models.model_trainer import CNNModel, ANNModel

logger = setup_logger()

class ModelManager:
    def __init__(self):
        self.models_dir = Config.MODELS_DIR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models = {}  # Cache for loaded models
    
    async def predict(self, model_name: str, user_name: str, 
                     data: Union[List[List[float]], Dict[str, Any]]) -> Dict[str, Any]:
        try:
            model_info = await self.load_model(model_name, user_name)
            model = model_info['model']
            preprocessors = model_info['preprocessors']
            config = model_info['config']
            
            task_type = config.task_type
            model_type = config.model_type
            
            if isinstance(data, dict) and 'image_path' in data:
                # Image prediction
                predictions = await self._predict_image(model, data['image_path'], preprocessors, config)
            elif isinstance(data, list):
                # Tabular prediction
                predictions = await self._predict_tabular(model, data, preprocessors, config)
            else:
                raise ValueError("Unsupported data format for prediction")
            
            return {
                'model_name': model_name,
                'model_type': model_type,
                'task_type': task_type,
                'predictions': predictions,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
    
    async def _predict_tabular(self, model: Any, data: List[List[float]], 
                              preprocessors: Dict[str, Any], config: Dict[str, Any]) -> List[Any]:
        task_type = config.task_type
        
        # Convert to DataFrame
        feature_columns = preprocessors.get('feature_columns', [])
        if len(feature_columns) != len(data[0]):
            raise ValueError(f"Expected {len(feature_columns)} features, got {len(data[0])}")
        
        df = pd.DataFrame(data, columns=feature_columns)
        
        # Apply feature scaling
        if 'feature_scaler' in preprocessors:
            X_scaled = preprocessors['feature_scaler'].transform(df)
        else:
            X_scaled = df.values
        
        # Make predictions
        if isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                outputs = model(X_tensor)
                
                if task_type == 'classification':
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                    
                    # Decode labels if label encoder exists
                    if 'label_encoder' in preprocessors:
                        predictions = preprocessors['label_encoder'].inverse_transform(predictions)
                    
                    result = []
                    for i, pred in enumerate(predictions):
                        result.append({
                            'prediction': pred,
                            'probabilities': probabilities[i].cpu().numpy().tolist()
                        })
                    return result
                else:
                    predictions = outputs.cpu().numpy().flatten()
                    
                    # Inverse transform if target scaler exists
                    if 'target_scaler' in preprocessors:
                        predictions = preprocessors['target_scaler'].inverse_transform(
                            predictions.reshape(-1, 1)
                        ).flatten()
                    
                    return predictions.tolist()
        
        else:  # Sklearn model
            predictions = model.predict(X_scaled)
            
            if task_type == 'classification':
                # Get prediction probabilities if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_scaled)
                    result = []
                    for i, pred in enumerate(predictions):
                        result.append({
                            'prediction': pred,
                            'probabilities': probabilities[i].tolist()
                        })
                    return result
                else:
                    return [{'prediction': pred} for pred in predictions.tolist()]
            else:
                # Inverse transform for regression if needed
                if 'target_scaler' in preprocessors:
                    predictions = preprocessors['target_scaler'].inverse_transform(
                        predictions.reshape(-1, 1)
                    ).flatten()
                
                return predictions.tolist()
    
    async def _predict_image(self, model: Any, image_path: str, 
                           preprocessors: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        model_type = config.model_type
        
        if model_type == 'YOLO':
            # YOLO prediction
            results = model(image_path)
            return {
                'detections': results[0].boxes.data.tolist() if results[0].boxes else [],
                'class_names': results[0].names if hasattr(results[0], 'names') else []
            }
        
        else:
            # CNN/ResNet prediction
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            model.eval()
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Decode class name if label encoder exists
            if 'label_encoder' in preprocessors:
                class_name = preprocessors['label_encoder'].inverse_transform([predicted_class])[0]
            else:
                class_name = str(predicted_class)
            
            return {
                'predicted_class': class_name,
                'confidence': float(confidence),
                'all_probabilities': probabilities[0].cpu().numpy().tolist()
            }
    
    async def load_model(self, model_name: str, user_name: str) -> Dict[str, Any]:
        cache_key = f"{user_name}_{model_name}"
        
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        user_dir = Config.get_user_dir(self.models_dir, user_name)
        model_files = list(user_dir.glob(f"{model_name}_{user_name}.*"))
        
        if not model_files:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_file = model_files[0]
        
        if model_file.suffix == '.pth':
            # PyTorch model
            checkpoint = torch.load(model_file, map_location=self.device)
            model_config = checkpoint['model_config']
            
            if model_config.model_type == 'CNN':
                # Reconstruct CNN model
                num_classes = len(checkpoint['preprocessors'].get('label_encoder', {}).get('classes_', [1]))
                if num_classes <= 1:
                    num_classes = checkpoint['metadata'].get('target_info', {}).get('num_classes', 2)
                
                depth = model_config.parameters.get('depth', 3)
                model = CNNModel(num_classes=num_classes, depth=depth)
                
            elif model_config.model_type in ['ANN', 'ANNRegressor']:
                # Reconstruct ANN model
                input_size = len(checkpoint['preprocessors'].get('feature_columns', []))
                if model_config.task_type == 'classification':
                    output_size = len(checkpoint['preprocessors'].get('label_encoder', {}).get('classes_', [2]))
                else:
                    output_size = 1
                
                depth = model_config.parameters.get('depth', 3)
                hidden_size = model_config.parameters.get('hidden_size', 128)
                model = ANNModel(input_size, output_size, depth, hidden_size, model_config.task_type)
            
            else:
                raise ValueError(f"Unsupported PyTorch model type: {model_config.model_type}")
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            model_info = {
                'model': model,
                'preprocessors': checkpoint['preprocessors'],
                'config': model_config,
                'metadata': checkpoint['metadata'],
                'metrics': checkpoint.get('metrics', {})
            }
        
        elif model_file.suffix == '.pt':
            # YOLO model
            from ultralytics import YOLO
            model = YOLO(str(model_file))
            
            # Load metadata
            metadata_file = user_dir / f"{model_name}_{user_name}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_info = json.load(f)
            else:
                metadata_info = {'model_config': {'model_type': 'YOLO', 'task_type': 'detection'}}
            
            model_info = {
                'model': model,
                'preprocessors': {},
                'config': metadata_info['model_config'],
                'metadata': metadata_info.get('metadata', {}),
                'metrics': metadata_info.get('metrics', {})
            }
        
        elif model_file.suffix == '.pkl':
            # Sklearn model
            model_data = joblib.load(model_file)
            
            model_info = {
                'model': model_data['model'],
                'preprocessors': model_data['preprocessors'],
                'config': model_data['model_config'],
                'metadata': model_data['metadata'],
                'metrics': model_data.get('metrics', {})
            }
        
        else:
            raise ValueError(f"Unsupported model file format: {model_file.suffix}")
        
        # Cache the loaded model
        self.loaded_models[cache_key] = model_info
        
        return model_info
    
    async def evaluate_model(self, model_name: str, user_name: str, 
                           test_data: UploadFile, viz_manager) -> Dict[str, Any]:
        try:
            model_info = await self.load_model(model_name, user_name)
            model = model_info['model']
            config = model_info['config']
            preprocessors = model_info['preprocessors']
            
            # Load test data
            if test_data.filename.endswith('.csv'):
                df = pd.read_csv(test_data.file)
            elif test_data.filename.endswith('.xlsx'):
                df = pd.read_excel(test_data.file)
            else:
                raise ValueError("Unsupported test data format")
            
            # Assume last column is target
            X_test = df.iloc[:, :-1]
            y_test = df.iloc[:, -1]
            
            # Preprocess test data
            if 'feature_scaler' in preprocessors:
                X_test_scaled = preprocessors['feature_scaler'].transform(X_test)
            else:
                X_test_scaled = X_test.values
            
            # Encode target if needed
            if config.task_type == 'classification' and 'label_encoder' in preprocessors:
                y_test_encoded = preprocessors['label_encoder'].transform(y_test)
            else:
                y_test_encoded = y_test.values
            
            # Make predictions
            if isinstance(model, nn.Module):
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
                    outputs = model(X_tensor)
                    
                    if config.task_type == 'classification':
                        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
                    else:
                        y_pred = outputs.cpu().numpy().flatten()
            else:
                y_pred = model.predict(X_test_scaled)
            
            # Generate evaluation visualizations and metrics
            eval_results = await viz_manager.generate_evaluation_visualizations(
                y_test_encoded, y_pred, model_name, user_name, config.task_type
            )
            
            return {
                'model_name': model_name,
                'evaluation_metrics': eval_results['metrics'],
                'visualizations': eval_results['visualizations'],
                'test_samples': len(y_test),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Evaluation failed: {str(e)}")
    
    async def list_user_models(self, user_name: str) -> Dict[str, Any]:
        user_dir = Config.get_user_dir(self.models_dir, user_name)
        
        if not user_dir.exists():
            return {"user_name": user_name, "models": []}
        
        models = []
        model_files = list(user_dir.glob("*.pth")) + list(user_dir.glob("*.pkl")) + list(user_dir.glob("*.pt"))
        
        for model_file in model_files:
            if model_file.name.endswith('_metadata.json'):
                continue
                
            try:
                # Extract model name (remove user_name suffix and extension)
                base_name = model_file.stem
                if f"_{user_name}" in base_name:
                    model_name = base_name.replace(f"_{user_name}", "")
                else:
                    model_name = base_name
                
                # Get model info
                if model_file.suffix == '.pth':
                    checkpoint = torch.load(model_file, map_location='cpu')
                    model_info = {
                        'model_name': model_name,
                        'model_type': checkpoint['model_config']['model_type'],
                        'task_type': checkpoint['model_config']['task_type'],
                        'metrics': checkpoint.get('metrics', {}),
                        'file_size': model_file.stat().st_size,
                        'created_at': model_file.stat().st_mtime
                    }
                elif model_file.suffix == '.pkl':
                    model_data = joblib.load(model_file)
                    model_info = {
                        'model_name': model_name,
                        'model_type': model_data['model_config']['model_type'],
                        'task_type': model_data['model_config']['task_type'],
                        'metrics': model_data.get('metrics', {}),
                        'file_size': model_file.stat().st_size,
                        'created_at': model_file.stat().st_mtime
                    }
                elif model_file.suffix == '.pt':
                    # YOLO model
                    metadata_file = user_dir / f"{base_name}_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        model_config = metadata.get('model_config', {})
                    else:
                        model_config = {'model_type': 'YOLO', 'task_type': 'detection'}
                    
                    model_info = {
                        'model_name': model_name,
                        'model_type': model_config.get('model_type', 'YOLO'),
                        'task_type': model_config.get('task_type', 'detection'),
                        'metrics': metadata.get('metrics', {}) if 'metadata' in locals() else {},
                        'file_size': model_file.stat().st_size,
                        'created_at': model_file.stat().st_mtime
                    }
                
                models.append(model_info)
                
            except Exception as e:
                logger.warning(f"Error reading model info for {model_file}: {str(e)}")
        
        return {"user_name": user_name, "models": models}
    
    async def get_model_info(self, model_name: str, user_name: str) -> Dict[str, Any]:
        try:
            model_info = await self.load_model(model_name, user_name)
            
            return {
                'model_name': model_name,
                'user_name': user_name,
                'model_type': model_info['config']['model_type'],
                'task_type': model_info['config']['task_type'],
                'parameters': model_info['config'].get('parameters', {}),
                'metrics': model_info['metrics'],
                'metadata': model_info['metadata'],
                'preprocessors': {k: str(type(v)) for k, v in model_info['preprocessors'].items()},
                'status': 'loaded'
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Model info not found: {str(e)}")
    
    async def get_model_path(self, model_name: str, user_name: str) -> Path:
        user_dir = Config.get_user_dir(self.models_dir, user_name)
        model_files = list(user_dir.glob(f"{model_name}_{user_name}.*"))
        
        if not model_files:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Create a zip file with model and metadata
        zip_path = user_dir / f"{model_name}_{user_name}.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for model_file in model_files:
                if not model_file.name.endswith('.zip'):
                    zip_file.write(model_file, model_file.name)
            
            # Include metadata file if it exists
            metadata_file = user_dir / f"{model_name}_{user_name}_metadata.json"
            if metadata_file.exists():
                zip_file.write(metadata_file, metadata_file.name)
        
        return zip_path
    
    async def delete_model(self, model_name: str, user_name: str):
        user_dir = Config.get_user_dir(self.models_dir, user_name)
        model_files = list(user_dir.glob(f"{model_name}_{user_name}.*"))
        
        if not model_files:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Remove from cache
        cache_key = f"{user_name}_{model_name}"
        if cache_key in self.loaded_models:
            del self.loaded_models[cache_key]
        
        # Delete all model files
        for model_file in model_files:
            if model_file.is_file():
                model_file.unlink()
        
        # Delete metadata file if exists
        metadata_file = user_dir / f"{model_name}_{user_name}_metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()
        
        logger.info(f"Model {model_name} deleted for user {user_name}")
    
    def clear_cache(self):
        """Clear the model cache to free memory"""
        self.loaded_models.clear()
        logger.info("Model cache cleared")