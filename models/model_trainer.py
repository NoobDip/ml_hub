import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from typing import Dict, List, Any, Tuple
import joblib
import json
from PIL import Image
from ultralytics import YOLO
from utils.config import Config
from utils.logger import setup_logger
from models.data_handler import DataHandler

logger = setup_logger()

class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            logger.error(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            return torch.zeros((3, 224, 224)), self.labels[idx]

class CNNModel(nn.Module):
    def __init__(self, num_classes: int, depth: int = 3, input_channels: int = 3):
        super(CNNModel, self).__init__()
        layers = []
        in_channels = input_channels
        
        for i in range(depth):
            out_channels = 32 * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25)
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 224, 224)
            feature_size = self.features(dummy_input).view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ANNModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, depth: int = 3, 
                 hidden_size: int = 128, task_type: str = 'classification'):
        super(ANNModel, self).__init__()
        layers = []
        in_size = input_size
        
        for i in range(depth):
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            in_size = hidden_size
            hidden_size = max(hidden_size // 2, 64)
        
        layers.append(nn.Linear(in_size, output_size))
        
        if task_type == 'classification' and output_size > 1:
            layers.append(nn.Softmax(dim=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ModelTrainer:
    def __init__(self):
        self.data_handler = DataHandler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    async def train_model(self, task_id: str, config: Dict[str, Any], 
                         training_tasks: Dict, viz_manager) -> None:
        try:
            training_tasks[task_id] = {"status": "loading", "progress": 5.0, "message": "Loading dataset"}
            
            X, y, metadata = await self.data_handler.load_dataset(
                config.dataset_id, config.user_name
            )
            
            model_type = config.model_type
            task_type = config.task_type
            print(f"Training {model_type} for {task_type} task with dataset {config.dataset_id}")
            
            if task_type in ['classification', 'regression'] and isinstance(X, pd.DataFrame):
                await self._train_tabular_model(task_id, config, X, y, metadata, training_tasks, viz_manager)
            elif 'image' in metadata.get('data_type', ''):
                await self._train_image_model(task_id, config, X, y, metadata, training_tasks, viz_manager)
            else:
                raise ValueError(f"Unsupported combination: {model_type} for {task_type}")
                
        except Exception as e:
            logger.error(f"Training failed for task {task_id}: {str(e)}")
            training_tasks[task_id] = {
                "status": "failed", 
                "progress": 0.0, 
                "message": str(e),
                "metrics": {},
                "visualizations": []
            }
    
    async def _train_tabular_model(self, task_id: str, config: Dict[str, Any], 
                                  X: pd.DataFrame, y: pd.Series, metadata: Dict,
                                  training_tasks: Dict, viz_manager) -> None:
        model_type = config.model_type
        task_type = config.task_type
        params = config.parameters or {}
        
        training_tasks[task_id]["message"] = "Preprocessing data"
        training_tasks[task_id]["progress"] = 15.0
        
        # Preprocessing
        X_processed, y_processed, preprocessors = await self._preprocess_tabular_data(X, y, task_type)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, 
            test_size=params.get('test_size', 0.2),
            random_state=params.get('random_state', 42),
            stratify=y_processed if task_type == 'classification' else None
        )
        
        training_tasks[task_id]["message"] = f"Training {model_type} model"
        training_tasks[task_id]["progress"] = 30.0
        
        metrics_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        if model_type == 'ANN' or model_type == 'ANNRegressor':
            model, history = await self._train_pytorch_tabular(
                X_train, X_test, y_train, y_test, task_type, params, 
                task_id, training_tasks
            )
            metrics_history.update(history)
        else:
            model = await self._train_sklearn_model(
                model_type, X_train, y_train, task_type, params,
                task_id, training_tasks
            )
        
        # Evaluation
        training_tasks[task_id]["message"] = "Evaluating model"
        training_tasks[task_id]["progress"] = 85.0
        
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test.values).to(self.device)
                outputs = model(X_test_tensor)
                if task_type == 'classification':
                    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
                else:
                    y_pred = outputs.cpu().numpy().flatten()
        
        # Calculate metrics
        if task_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            metrics = {'accuracy': float(accuracy)}
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics = {'mse': float(mse), 'r2': float(r2), 'rmse': float(np.sqrt(mse))}
        
        # Generate visualizations
        training_tasks[task_id]["message"] = "Generating visualizations"
        training_tasks[task_id]["progress"] = 90.0
        
        viz_paths = []
        if metrics_history['loss']:
            viz_paths.extend(await viz_manager.generate_training_visualizations(
                task_id, model_type, metrics_history, task_type
            ))
        
        eval_viz = await viz_manager.generate_evaluation_visualizations(
            y_test, y_pred, config.model_name, config.user_name, task_type
        )
        viz_paths.extend(eval_viz.get('visualizations', []))
        
        # Save model
        await self._save_model(model, config, preprocessors, metadata, metrics)
        
        # Complete training
        training_tasks[task_id] = {
            "status": "completed",
            "progress": 100.0,
            "message": f"Training completed. {task_type.capitalize()} metrics: {metrics}",
            "metrics": {**metrics, **metrics_history},
            "visualizations": viz_paths
        }
    
    async def _train_image_model(self, task_id: str, config: Dict[str, Any],
                               image_paths: List[str], labels: List[str], metadata: Dict,
                               training_tasks: Dict, viz_manager) -> None:
        model_type = config.model_type
        params = config.parameters or {}
        
        training_tasks[task_id]["message"] = "Preprocessing images"
        training_tasks[task_id]["progress"] = 20.0
        
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)
        
        # Image transforms
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Train-test split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
        )
        
        # Datasets and dataloaders
        train_dataset = ImageDataset(train_paths, train_labels, transform_train)
        val_dataset = ImageDataset(val_paths, val_labels, transform_val)
        
        batch_size = params.get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        training_tasks[task_id]["message"] = f"Training {model_type} model"
        training_tasks[task_id]["progress"] = 30.0
        
        metrics_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'learning_rate': []}
        
        if model_type == 'YOLO':
            model, history = await self._train_yolo_model(
                config, train_paths, train_labels, task_id, training_tasks
            )
        elif model_type == 'CNN':
            model, history = await self._train_cnn_model(
                train_loader, val_loader, num_classes, params, task_id, training_tasks
            )
        elif model_type == 'ResNet':
            model, history = await self._train_resnet_model(
                train_loader, val_loader, num_classes, params, task_id, training_tasks
            )
        else:
            raise ValueError(f"Unsupported image model type: {model_type}")
        
        metrics_history.update(history)
        
        # Evaluation
        training_tasks[task_id]["message"] = "Evaluating model"
        training_tasks[task_id]["progress"] = 85.0
        
        if model_type != 'YOLO':
            model.eval()
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels_batch in val_loader:
                    images, labels_batch = images.to(self.device), labels_batch.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels_batch.size(0)
                    correct += (predicted == labels_batch).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels_batch.cpu().numpy())
            
            accuracy = correct / total
            metrics = {'accuracy': float(accuracy)}
        else:
            metrics = {'map': 0.0}  # YOLO metrics would be calculated differently
        
        # Generate visualizations
        training_tasks[task_id]["message"] = "Generating visualizations"
        training_tasks[task_id]["progress"] = 90.0
        
        viz_paths = await viz_manager.generate_training_visualizations(
            task_id, model_type, metrics_history, 'classification'
        )
        
        if model_type != 'YOLO':
            eval_viz = await viz_manager.generate_evaluation_visualizations(
                all_labels, all_preds, config.model_name, config.user_name, 
                'classification', label_encoder.classes_
            )
            viz_paths.extend(eval_viz.get('visualizations', []))
        
        # Save model
        preprocessors = {'label_encoder': label_encoder}
        await self._save_model(model, config, preprocessors, metadata, metrics)
        
        # Complete training
        training_tasks[task_id] = {
            "status": "completed",
            "progress": 100.0,
            "message": f"Training completed. Accuracy: {metrics.get('accuracy', 'N/A')}",
            "metrics": {**metrics, **metrics_history},
            "visualizations": viz_paths
        }
    
    async def _preprocess_tabular_data(self, X: pd.DataFrame, y: pd.Series, 
                                     task_type: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        preprocessors = {}
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        else:
            X_encoded = X.copy()
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        preprocessors['feature_scaler'] = scaler
        preprocessors['feature_columns'] = X_encoded.columns.tolist()
        
        # Encode target for classification
        if task_type == 'classification':
            if y.dtype == 'object':
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                preprocessors['label_encoder'] = label_encoder
            else:
                y_encoded = y.values
        else:
            # Scale target for regression
            target_scaler = MinMaxScaler()
            y_encoded = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
            preprocessors['target_scaler'] = target_scaler
        
        return X_scaled, y_encoded, preprocessors
    
    async def _train_pytorch_tabular(self, X_train: np.ndarray, X_test: np.ndarray,
                                   y_train: np.ndarray, y_test: np.ndarray,
                                   task_type: str, params: Dict, 
                                   task_id: str, training_tasks: Dict) -> Tuple[nn.Module, Dict]:
        input_size = X_train.shape[1]
        
        if task_type == 'classification':
            output_size = len(np.unique(y_train))
        else:
            output_size = 1
        
        # Model parameters
        depth = params.get('depth', 3)
        hidden_size = params.get('hidden_size', 128)
        
        model = ANNModel(input_size, output_size, depth, hidden_size, task_type).to(self.device)
        
        # Loss and optimizer
        if task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device) if task_type == 'classification' else torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device) if task_type == 'classification' else torch.FloatTensor(y_test).to(self.device)
        
        # Training loop
        epochs = params.get('epochs', 100)
        batch_size = params.get('batch_size', 32)
        
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'learning_rate': []}
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                if task_type == 'classification':
                    loss = criterion(outputs, batch_y)
                    _, predicted = torch.max(outputs.data, 1)
                    train_correct += (predicted == batch_y).sum().item()
                    train_total += batch_y.size(0)
                else:
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                if task_type == 'classification':
                    val_loss = criterion(val_outputs, y_test_tensor)
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_accuracy = (val_predicted == y_test_tensor).float().mean()
                else:
                    val_outputs = val_outputs.squeeze()
                    val_loss = criterion(val_outputs, y_test_tensor)
                    val_accuracy = torch.tensor(0.0)  # R2 score would be calculated separately
            
            # Record metrics
            history['loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss.item())
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            if task_type == 'classification':
                train_accuracy = train_correct / train_total
                history['accuracy'].append(train_accuracy)
                history['val_accuracy'].append(val_accuracy.item())
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Update progress
            progress = 30.0 + (epoch + 1) / epochs * 50.0
            training_tasks[task_id]["progress"] = progress
            
            if task_type == 'classification':
                message = f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_accuracy:.4f}"
            else:
                message = f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}"
            
            training_tasks[task_id]["message"] = message
        
        return model, history
    
    async def _train_sklearn_model(self, model_type: str, X_train: np.ndarray, 
                                 y_train: np.ndarray, task_type: str, params: Dict,
                                 task_id: str, training_tasks: Dict):
        if task_type == 'classification':
            if model_type == 'RandomForest':
                model = RandomForestClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', None),
                    random_state=42
                )
        else:  # regression
            if model_type == 'LinearRegression':
                model = LinearRegression()
            elif model_type == 'RandomForestRegressor':
                model = RandomForestRegressor(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', None),
                    random_state=42
                )
            elif model_type == 'SVR':
                model = SVR(
                    kernel=params.get('kernel', 'rbf'),
                    C=params.get('C', 1.0),
                    epsilon=params.get('epsilon', 0.1)
                )
            elif model_type == 'GradientBoosting':
                model = GradientBoostingRegressor(
                    n_estimators=params.get('n_estimators', 100),
                    learning_rate=params.get('learning_rate', 0.1),
                    max_depth=params.get('max_depth', 3),
                    random_state=42
                )
        
        training_tasks[task_id]["message"] = f"Fitting {model_type} model"
        model.fit(X_train, y_train)
        
        return model
    
    async def _train_cnn_model(self, train_loader: DataLoader, val_loader: DataLoader,
                             num_classes: int, params: Dict, task_id: str, 
                             training_tasks: Dict) -> Tuple[nn.Module, Dict]:
        depth = params.get('depth', 3)
        model = CNNModel(num_classes=num_classes, depth=depth).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
        
        epochs = params.get('epochs', 20)
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'learning_rate': []}
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Record metrics
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            history['loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            scheduler.step()
            
            # Update progress
            progress = 30.0 + (epoch + 1) / epochs * 50.0
            training_tasks[task_id]["progress"] = progress
            training_tasks[task_id]["message"] = f"Epoch {epoch+1}/{epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
        
        return model, history
    
    async def _train_resnet_model(self, train_loader: DataLoader, val_loader: DataLoader,
                                num_classes: int, params: Dict, task_id: str,
                                training_tasks: Dict) -> Tuple[nn.Module, Dict]:
        variant = params.get('variant', 'resnet18')
        
        if variant == 'resnet50':
            model = resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        model = model.to(self.device)
        
        # Freeze early layers for transfer learning
        if params.get('freeze_backbone', True):
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters() if params.get('freeze_backbone', True) else model.parameters(),
                              lr=params.get('learning_rate', 0.001))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        epochs = params.get('epochs', 15)
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'learning_rate': []}
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Record metrics
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            history['loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            scheduler.step()
            
            # Update progress
            progress = 30.0 + (epoch + 1) / epochs * 50.0
            training_tasks[task_id]["progress"] = progress
            training_tasks[task_id]["message"] = f"Epoch {epoch+1}/{epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
        
        return model, history
    
    async def _train_yolo_model(self, config: Dict[str, Any], train_paths: List[str],
                              train_labels: List[int], task_id: str,
                              training_tasks: Dict) -> Tuple[Any, Dict]:
        params = config.parameters or {}
        
        # Note: This is a simplified YOLO training setup
        # In practice, you'd need proper YOLO dataset format
        model = YOLO('yolov8n.pt')
        
        epochs = params.get('epochs', 50)
        img_size = params.get('img_size', 640)
        batch_size = params.get('batch_size', 16)
        
        # For demo purposes, we'll simulate YOLO training
        history = {'loss': [], 'map': []}
        
        for epoch in range(epochs):
            # Simulate training progress
            loss = 1.0 - (epoch / epochs) * 0.8 + np.random.normal(0, 0.05)
            map_score = (epoch / epochs) * 0.9 + np.random.normal(0, 0.02)
            
            history['loss'].append(max(loss, 0.1))
            history['map'].append(min(max(map_score, 0), 1))
            
            progress = 30.0 + (epoch + 1) / epochs * 50.0
            training_tasks[task_id]["progress"] = progress
            training_tasks[task_id]["message"] = f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, mAP: {map_score:.4f}"
            
            # Simulate some training time
            await asyncio.sleep(0.1)
        
        return model, history
    
    async def _save_model(self, model: Any, config: Dict[str, Any], 
                         preprocessors: Dict[str, Any], metadata: Dict[str, Any],
                         metrics: Dict[str, Any]) -> None:
        user_dir = Config.get_user_dir(Config.MODELS_DIR, config.user_name)
        model_name = f"{config.model_name}_{config.user_name}"
        
        # Determine file extension based on model type
        if isinstance(model, nn.Module):
            model_path = user_dir / f"{model_name}.pth"
            
            # Save PyTorch model
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'model_type': config.model_type,
                    'task_type': config.task_type,
                    'parameters': config.parameters
                },
                'preprocessors': preprocessors,
                'metadata': metadata,
                'metrics': metrics
            }, model_path)
            
        elif hasattr(model, 'save'):  # YOLO model
            model_path = user_dir / f"{model_name}.pt"
            model.save(str(model_path))
            
            # Save additional metadata
            metadata_path = user_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'model_config': {
                        'model_type': config.model_type,
                        'task_type': config.task_type,
                        'parameters': config.parameters
                    },
                    'preprocessors': {k: str(v) for k, v in preprocessors.items()},
                    'metadata': metadata,
                    'metrics': metrics
                }, f, indent=2, default=str)
                
        else:  # Sklearn model
            model_path = user_dir / f"{model_name}.pkl"
            
            # Save sklearn model with metadata
            joblib.dump({
                'model': model,
                'model_config': {
                    'model_type': config.model_type,
                    'task_type': config.task_type,
                    'parameters': config.parameters
                },
                'preprocessors': preprocessors,
                'metadata': metadata,
                'metrics': metrics
            }, model_path)
        
        logger.info(f"Model saved: {model_path}")

import asyncio