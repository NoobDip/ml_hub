import os
import shutil
import uuid
import pandas as pd
import numpy as np
import zipfile
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import UploadFile, HTTPException
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger()

class DatasetInfo:
    def __init__(self, dataset_id: str, data_type: str, task_type: str, 
                 shape: tuple, columns: Optional[List[str]] = None,
                 target_info: Dict[str, Any] = None, statistics: Dict[str, Any] = None):
        self.dataset_id = dataset_id
        self.data_type = data_type
        self.task_type = task_type
        self.shape = shape
        self.columns = columns or []
        self.target_info = target_info or {}
        self.statistics = statistics or {}

class DataHandler:
    def __init__(self):
        self.upload_dir = Config.UPLOAD_DIR
        self.temp_dir = Config.TEMP_DIR
    
    async def upload_dataset(self, file: UploadFile, data_type: str, 
                           task_type: str, user_name: str, 
                           target_column: Optional[str] = None) -> DatasetInfo:
        dataset_id = str(uuid.uuid4())
        user_dir = Config.get_user_dir(self.upload_dir, user_name)
        
        if file.size > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        file_path = user_dir / f"{dataset_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if data_type == "tabular":
            return await self._process_tabular_data(
                file_path, dataset_id, task_type, target_column
            )
        elif data_type == "image":
            return await self._process_image_data(
                file_path, dataset_id, task_type, user_name
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported data type")
    
    async def _process_tabular_data(self, file_path: Path, dataset_id: str, 
                                  task_type: str, target_column: Optional[str]) -> DatasetInfo:
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported tabular file format")
            
            if df.empty:
                raise ValueError("Dataset is empty")
            
            target_col = target_column or df.columns[-1]
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found")
            
            target_info = self._analyze_target_column(df[target_col], task_type)
            statistics = self._generate_dataset_statistics(df)
            
            metadata = {
                'target_column': target_col,
                'task_type': task_type,
                'file_path': str(file_path)
            }
            
            metadata_path = file_path.with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            return DatasetInfo(
                dataset_id=dataset_id,
                data_type="tabular",
                task_type=task_type,
                shape=df.shape,
                columns=df.columns.tolist(),
                target_info=target_info,
                statistics=statistics
            )
            
        except Exception as e:
            logger.error(f"Error processing tabular data: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")
    
    async def _process_image_data(self, file_path: Path, dataset_id: str, 
                                task_type: str, user_name: str) -> DatasetInfo:
        try:
            if not file_path.suffix.lower() == '.zip':
                raise ValueError("Image datasets must be uploaded as ZIP files")
            
            extract_dir = Config.get_user_dir(self.temp_dir, user_name) / dataset_id
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            image_files = []
            classes = set()
            
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in Config.SUPPORTED_IMAGE_FORMATS):
                        image_path = Path(root) / file
                        image_files.append(str(image_path))
                        
                        class_name = Path(root).name
                        if class_name != dataset_id:
                            classes.add(class_name)
            
            if not image_files:
                raise ValueError("No valid image files found in ZIP")
            
            sample_image = Image.open(image_files[0])
            image_shape = (*sample_image.size, len(sample_image.getbands()))
            
            if task_type == "classification":
                if not classes:
                    raise ValueError("No class directories found for classification task")
                target_info = {
                    "num_classes": len(classes),
                    "classes": list(classes),
                    "type": "categorical"
                }
            else:
                target_info = {"type": "detection/segmentation"}
            
            statistics = {
                "total_images": len(image_files),
                "image_shape": image_shape,
                "classes_distribution": {cls: sum(1 for img in image_files if cls in img) for cls in classes}
            }
            
            metadata = {
                'task_type': task_type,
                'extract_dir': str(extract_dir),
                'file_path': str(file_path),
                'image_files': image_files[:100]
            }
            
            metadata_path = file_path.with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, default=str)
            
            return DatasetInfo(
                dataset_id=dataset_id,
                data_type="image",
                task_type=task_type,
                shape=(len(image_files), *image_shape),
                target_info=target_info,
                statistics=statistics
            )
            
        except Exception as e:
            logger.error(f"Error processing image data: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing images: {str(e)}")
    
    def _analyze_target_column(self, target_series: pd.Series, task_type: str) -> Dict[str, Any]:
        if task_type == "classification":
            unique_values = target_series.unique()
            return {
                "type": "categorical",
                "num_classes": len(unique_values),
                "classes": unique_values.tolist(),
                "distribution": target_series.value_counts().to_dict()
            }
        elif task_type == "regression":
            return {
                "type": "continuous",
                "min": float(target_series.min()),
                "max": float(target_series.max()),
                "mean": float(target_series.mean()),
                "std": float(target_series.std()),
                "median": float(target_series.median())
            }
        else:
            return {"type": "unknown"}
    
    def _generate_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        stats = {
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        if len(numeric_cols) > 0:
            stats["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        if len(categorical_cols) > 0:
            stats["categorical_summary"] = {
                col: df[col].value_counts().head(10).to_dict() 
                for col in categorical_cols
            }
        
        return stats
    
    async def get_dataset_info(self, dataset_id: str, user_name: str) -> Dict[str, Any]:
        user_dir = Config.get_user_dir(self.upload_dir, user_name)
        dataset_files = list(user_dir.glob(f"{dataset_id}_*"))
        
        if not dataset_files:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        metadata_files = [f for f in dataset_files if f.suffix == '.json']
        if not metadata_files:
            raise HTTPException(status_code=404, detail="Dataset metadata not found")
        
        import json
        with open(metadata_files[0], 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    async def list_user_datasets(self, user_name: str) -> Dict[str, Any]:
        user_dir = Config.get_user_dir(self.upload_dir, user_name)
        
        if not user_dir.exists():
            return {"user_name": user_name, "datasets": []}
        
        datasets = []
        metadata_files = list(user_dir.glob("*.json"))
        
        for metadata_file in metadata_files:
            try:
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                dataset_id = metadata_file.stem.split('_')[0]
                datasets.append({
                    "dataset_id": dataset_id,
                    "task_type": metadata.get("task_type"),
                    "uploaded_at": metadata_file.stat().st_mtime
                })
            except Exception as e:
                logger.warning(f"Error reading metadata file {metadata_file}: {str(e)}")
        
        return {"user_name": user_name, "datasets": datasets}
    
    async def delete_dataset(self, dataset_id: str, user_name: str):
        user_dir = Config.get_user_dir(self.upload_dir, user_name)
        dataset_files = list(user_dir.glob(f"{dataset_id}_*"))
        
        if not dataset_files:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        for file_path in dataset_files:
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
        
        temp_dir = Config.get_user_dir(self.temp_dir, user_name) / dataset_id
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        logger.info(f"Dataset {dataset_id} deleted for user {user_name}")
    
    async def load_dataset(self, dataset_id: str, user_name: str) -> tuple:
        metadata = await self.get_dataset_info(dataset_id, user_name)
        
        if 'file_path' not in metadata:
            raise ValueError("Dataset file path not found in metadata")
        
        file_path = Path(metadata['file_path'])
        
        if not file_path.exists():
            raise ValueError("Dataset file not found")
        
        if metadata.get('task_type') in ['classification', 'regression']:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            target_col = metadata.get('target_column', df.columns[-1])
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            return X, y, metadata
        
        else:
            extract_dir = Path(metadata['extract_dir'])
            image_files = []
            labels = []
            
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in Config.SUPPORTED_IMAGE_FORMATS):
                        image_files.append(str(Path(root) / file))
                        labels.append(Path(root).name)
            
            return image_files, labels, metadata