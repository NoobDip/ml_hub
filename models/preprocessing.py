import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import joblib
from PIL import Image, ImageEnhance, ImageFilter
from utils.config import Config
from utils.logger import setup_logger
from models.data_handler import DataHandler

logger = setup_logger()

class PreprocessingManager:
    def __init__(self):
        self.data_handler = DataHandler()
        self.preprocessors = {}
    
    async def preprocess_dataset(self, dataset_id: str, user_name: str, 
                               config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            X, y, metadata = await self.data_handler.load_dataset(dataset_id, user_name)
            
            if isinstance(X, pd.DataFrame):
                return await self._preprocess_tabular(X, y, metadata, config, dataset_id, user_name)
            else:
                return await self._preprocess_images(X, y, metadata, config, dataset_id, user_name)
                
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise Exception(f"Preprocessing failed: {str(e)}")
    
    async def _preprocess_tabular(self, X: pd.DataFrame, y: pd.Series, metadata: Dict,
                                config: Dict[str, Any], dataset_id: str, user_name: str) -> Dict[str, Any]:
        preprocessing_steps = []
        X_processed = X.copy()
        
        # Handle missing values
        missing_strategy = config.get('missing_values', 'simple')
        if X_processed.isnull().sum().sum() > 0:
            X_processed, imputer = await self._handle_missing_values(X_processed, missing_strategy)
            preprocessing_steps.append({
                'step': 'imputation',
                'method': missing_strategy,
                'affected_columns': X.columns[X.isnull().any()].tolist()
            })
            self.preprocessors[f'{dataset_id}_imputer'] = imputer
        
        # Handle outliers
        outlier_method = config.get('outliers', 'none')
        if outlier_method != 'none':
            X_processed, outlier_mask = await self._handle_outliers(X_processed, outlier_method)
            preprocessing_steps.append({
                'step': 'outlier_removal',
                'method': outlier_method,
                'outliers_removed': int(outlier_mask.sum())
            })
        
        # Feature encoding
        encoding_method = config.get('encoding', 'onehot')
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X_processed, encoders = await self._encode_categorical(X_processed, categorical_cols, encoding_method)
            preprocessing_steps.append({
                'step': 'categorical_encoding',
                'method': encoding_method,
                'encoded_columns': categorical_cols.tolist()
            })
            self.preprocessors[f'{dataset_id}_encoders'] = encoders
        
        # Feature scaling
        scaling_method = config.get('scaling', 'standard')
        if scaling_method != 'none':
            X_processed, scaler = await self._scale_features(X_processed, scaling_method)
            preprocessing_steps.append({
                'step': 'feature_scaling',
                'method': scaling_method,
                'scaled_features': X_processed.shape[1]
            })
            self.preprocessors[f'{dataset_id}_scaler'] = scaler
        
        # Feature selection
        feature_selection = config.get('feature_selection', {})
        if feature_selection.get('method', 'none') != 'none':
            X_processed, selector = await self._select_features(
                X_processed, y, feature_selection, metadata.get('task_type', 'classification')
            )
            preprocessing_steps.append({
                'step': 'feature_selection',
                'method': feature_selection['method'],
                'features_selected': X_processed.shape[1],
                'features_removed': X.shape[1] - X_processed.shape[1]
            })
            self.preprocessors[f'{dataset_id}_selector'] = selector
        
        # Dimensionality reduction
        dim_reduction = config.get('dimensionality_reduction', {})
        if dim_reduction.get('method', 'none') != 'none':
            X_processed, reducer = await self._reduce_dimensions(X_processed, dim_reduction)
            preprocessing_steps.append({
                'step': 'dimensionality_reduction',
                'method': dim_reduction['method'],
                'components': X_processed.shape[1]
            })
            self.preprocessors[f'{dataset_id}_reducer'] = reducer
        
        # Save preprocessing pipeline
        pipeline_path = await self._save_preprocessing_pipeline(
            dataset_id, user_name, preprocessing_steps, self.preprocessors
        )
        
        # Calculate preprocessing statistics
        preprocessing_stats = {
            'original_shape': X.shape,
            'processed_shape': X_processed.shape,
            'missing_values_before': int(X.isnull().sum().sum()),
            'missing_values_after': int(X_processed.isnull().sum().sum()),
            'categorical_columns': len(categorical_cols),
            'numerical_columns': len(X_processed.select_dtypes(include=[np.number]).columns)
        }
        
        return {
            'dataset_id': dataset_id,
            'preprocessing_steps': preprocessing_steps,
            'preprocessing_stats': preprocessing_stats,
            'pipeline_path': str(pipeline_path),
            'status': 'completed'
        }
    
    async def _preprocess_images(self, image_paths: List[str], labels: List[str], 
                               metadata: Dict, config: Dict[str, Any], 
                               dataset_id: str, user_name: str) -> Dict[str, Any]:
        preprocessing_steps = []
        processed_count = 0
        
        # Image preprocessing parameters
        target_size = config.get('target_size', (224, 224))
        augmentation = config.get('augmentation', {})
        
        user_temp_dir = Config.get_user_dir(Config.TEMP_DIR, user_name)
        processed_dir = user_temp_dir / f"processed_{dataset_id}"
        processed_dir.mkdir(exist_ok=True)
        
        processed_paths = []
        
        for i, img_path in enumerate(image_paths):
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')
                
                # Resize
                if image.size != target_size:
                    image = image.resize(target_size, Image.Resampling.LANCZOS)
                
                # Apply augmentation if specified
                if augmentation.get('brightness', False):
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(np.random.uniform(0.8, 1.2))
                
                if augmentation.get('contrast', False):
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(np.random.uniform(0.8, 1.2))
                
                if augmentation.get('blur', False) and np.random.random() < 0.3:
                    image = image.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.5, 1.5)))
                
                # Save processed image
                processed_path = processed_dir / f"processed_{i}.jpg"
                image.save(processed_path, 'JPEG', quality=95)
                processed_paths.append(str(processed_path))
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Error processing image {img_path}: {str(e)}")
                processed_paths.append(img_path)  # Keep original if processing fails
        
        preprocessing_steps.append({
            'step': 'image_preprocessing',
            'target_size': target_size,
            'augmentation': augmentation,
            'processed_images': processed_count,
            'failed_images': len(image_paths) - processed_count
        })
        
        # Save preprocessing info
        preprocessing_info = {
            'original_paths': image_paths,
            'processed_paths': processed_paths,
            'labels': labels,
            'preprocessing_steps': preprocessing_steps
        }
        
        info_path = processed_dir / 'preprocessing_info.json'
        with open(info_path, 'w') as f:
            json.dump(preprocessing_info, f, indent=2)
        
        return {
            'dataset_id': dataset_id,
            'preprocessing_steps': preprocessing_steps,
            'processed_images': processed_count,
            'processed_dir': str(processed_dir),
            'status': 'completed'
        }
    
    async def _handle_missing_values(self, X: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, Any]:
        if strategy == 'simple':
            numeric_imputer = SimpleImputer(strategy='median')
            categorical_imputer = SimpleImputer(strategy='most_frequent')
        elif strategy == 'knn':
            numeric_imputer = KNNImputer(n_neighbors=5)
            categorical_imputer = SimpleImputer(strategy='most_frequent')
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")
        
        X_imputed = X.copy()
        
        # Handle numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_imputed[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X_imputed[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
        
        imputers = {
            'numeric': numeric_imputer if len(numeric_cols) > 0 else None,
            'categorical': categorical_imputer if len(categorical_cols) > 0 else None
        }
        
        return X_imputed, imputers
    
    async def _handle_outliers(self, X: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, np.ndarray]:
        if method == 'isolation_forest':
            clf = IsolationForest(contamination=0.1, random_state=42)
            outlier_mask = clf.fit_predict(X.select_dtypes(include=[np.number])) == -1
        elif method == 'iqr':
            outlier_mask = np.zeros(len(X), dtype=bool)
            for col in X.select_dtypes(include=[np.number]).columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask |= (X[col] < (Q1 - 1.5 * IQR)) | (X[col] > (Q3 + 1.5 * IQR))
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return X[~outlier_mask], outlier_mask
    
    async def _encode_categorical(self, X: pd.DataFrame, categorical_cols: pd.Index, 
                                method: str) -> Tuple[pd.DataFrame, Dict]:
        encoders = {}
        X_encoded = X.copy()
        
        if method == 'onehot':
            for col in categorical_cols:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_cols = encoder.fit_transform(X[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                
                # Remove original column and add encoded columns
                X_encoded = X_encoded.drop(columns=[col])
                for i, feature_name in enumerate(feature_names):
                    X_encoded[feature_name] = encoded_cols[:, i]
                
                encoders[col] = encoder
                
        elif method == 'label':
            for col in categorical_cols:
                encoder = LabelEncoder()
                X_encoded[col] = encoder.fit_transform(X[col])
                encoders[col] = encoder
        
        return X_encoded, encoders
    
    async def _scale_features(self, X: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, Any]:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_scaled = X.copy()
        
        if len(numeric_cols) > 0:
            X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        
        return X_scaled, scaler
    
    async def _select_features(self, X: pd.DataFrame, y: pd.Series, 
                             config: Dict[str, Any], task_type: str) -> Tuple[pd.DataFrame, Any]:
        method = config['method']
        k = config.get('k', 10)
        
        if task_type == 'classification':
            if method == 'f_classif':
                selector = SelectKBest(score_func=f_classif, k=k)
            elif method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
            else:
                raise ValueError(f"Unknown feature selection method: {method}")
        else:  # regression
            if method == 'f_regression':
                selector = SelectKBest(score_func=f_regression, k=k)
            else:
                raise ValueError(f"Unknown feature selection method: {method}")
        
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()]
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        return X_selected_df, selector
    
    async def _reduce_dimensions(self, X: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Any]:
        method = config['method']
        
        if method == 'pca':
            n_components = config.get('n_components', 0.95)  # Keep 95% variance by default
            reducer = PCA(n_components=n_components)
            X_reduced = reducer.fit_transform(X)
            
            # Create column names for PCA components
            component_names = [f'PC{i+1}' for i in range(X_reduced.shape[1])]
            X_reduced_df = pd.DataFrame(X_reduced, columns=component_names, index=X.index)
            
            return X_reduced_df, reducer
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    async def _save_preprocessing_pipeline(self, dataset_id: str, user_name: str,
                                         steps: List[Dict], preprocessors: Dict) -> Path:
        user_dir = Config.get_user_dir(Config.TEMP_DIR, user_name)
        pipeline_dir = user_dir / f"preprocessing_{dataset_id}"
        pipeline_dir.mkdir(exist_ok=True)
        
        # Save preprocessing steps
        steps_path = pipeline_dir / 'preprocessing_steps.json'
        with open(steps_path, 'w') as f:
            json.dump(steps, f, indent=2)
        
        # Save preprocessors
        for name, preprocessor in preprocessors.items():
            if preprocessor is not None:
                preprocessor_path = pipeline_dir / f'{name}.pkl'
                joblib.dump(preprocessor, preprocessor_path)
        
        return pipeline_dir
    
    async def apply_preprocessing_pipeline(self, dataset_id: str, user_name: str, 
                                        X: pd.DataFrame) -> pd.DataFrame:
        """Apply saved preprocessing pipeline to new data"""
        user_dir = Config.get_user_dir(Config.TEMP_DIR, user_name)
        pipeline_dir = user_dir / f"preprocessing_{dataset_id}"
        
        if not pipeline_dir.exists():
            raise ValueError(f"Preprocessing pipeline not found for dataset {dataset_id}")
        
        # Load preprocessing steps
        steps_path = pipeline_dir / 'preprocessing_steps.json'
        with open(steps_path, 'r') as f:
            steps = json.load(f)
        
        X_processed = X.copy()
        
        for step in steps:
            step_type = step['step']
            
            if step_type == 'imputation':
                # Load and apply imputers
                numeric_imputer_path = pipeline_dir / f'{dataset_id}_imputer.pkl'
                if numeric_imputer_path.exists():
                    imputers = joblib.load(numeric_imputer_path)
                    
                    numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
                    categorical_cols = X_processed.select_dtypes(include=['object']).columns
                    
                    if imputers['numeric'] and len(numeric_cols) > 0:
                        X_processed[numeric_cols] = imputers['numeric'].transform(X_processed[numeric_cols])
                    
                    if imputers['categorical'] and len(categorical_cols) > 0:
                        X_processed[categorical_cols] = imputers['categorical'].transform(X_processed[categorical_cols])
            
            elif step_type == 'categorical_encoding':
                # Load and apply encoders
                encoders_path = pipeline_dir / f'{dataset_id}_encoders.pkl'
                if encoders_path.exists():
                    encoders = joblib.load(encoders_path)
                    
                    for col, encoder in encoders.items():
                        if col in X_processed.columns:
                            if isinstance(encoder, OneHotEncoder):
                                encoded_cols = encoder.transform(X_processed[[col]])
                                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                                
                                X_processed = X_processed.drop(columns=[col])
                                for i, feature_name in enumerate(feature_names):
                                    X_processed[feature_name] = encoded_cols[:, i]
                            else:  # LabelEncoder
                                X_processed[col] = encoder.transform(X_processed[col])
            
            elif step_type == 'feature_scaling':
                # Load and apply scaler
                scaler_path = pipeline_dir / f'{dataset_id}_scaler.pkl'
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)
                    numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        X_processed[numeric_cols] = scaler.transform(X_processed[numeric_cols])
            
            elif step_type == 'feature_selection':
                # Load and apply selector
                selector_path = pipeline_dir / f'{dataset_id}_selector.pkl'
                if selector_path.exists():
                    selector = joblib.load(selector_path)
                    X_selected = selector.transform(X_processed)
                    selected_features = X_processed.columns[selector.get_support()]
                    X_processed = pd.DataFrame(X_selected, columns=selected_features, index=X_processed.index)
            
            elif step_type == 'dimensionality_reduction':
                # Load and apply reducer
                reducer_path = pipeline_dir / f'{dataset_id}_reducer.pkl'
                if reducer_path.exists():
                    reducer = joblib.load(reducer_path)
                    X_reduced = reducer.transform(X_processed)
                    component_names = [f'PC{i+1}' for i in range(X_reduced.shape[1])]
                    X_processed = pd.DataFrame(X_reduced, columns=component_names, index=X_processed.index)
        
        return X_processed
    
    def get_preprocessing_summary(self, dataset_id: str, user_name: str) -> Dict[str, Any]:
        """Get summary of preprocessing steps applied to a dataset"""
        user_dir = Config.get_user_dir(Config.TEMP_DIR, user_name)
        pipeline_dir = user_dir / f"preprocessing_{dataset_id}"
        
        if not pipeline_dir.exists():
            return {"error": "Preprocessing pipeline not found"}
        
        steps_path = pipeline_dir / 'preprocessing_steps.json'
        with open(steps_path, 'r') as f:
            steps = json.load(f)
        
        return {
            'dataset_id': dataset_id,
            'preprocessing_steps': steps,
            'pipeline_path': str(pipeline_dir)
        }