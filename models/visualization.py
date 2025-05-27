import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger()

class VisualizationManager:
    def __init__(self):
        self.viz_dir = Config.VIZ_DIR
        self.style = Config.VISUALIZATION_SETTINGS["style"]
        self.figure_size = Config.VISUALIZATION_SETTINGS["figure_size"]
        self.dpi = Config.VISUALIZATION_SETTINGS["dpi"]
        plt.style.use('default')
        sns.set_theme()
    
    async def generate_dataset_visualizations(self, dataset_id: str, user_name: str) -> List[str]:
        from models.data_handler import DataHandler
        data_handler = DataHandler()
        
        try:
            X, y, metadata = await data_handler.load_dataset(dataset_id, user_name)
            viz_paths = []
            
            user_viz_dir = Config.get_user_dir(self.viz_dir, user_name)
            dataset_viz_dir = user_viz_dir / f"dataset_{dataset_id}"
            dataset_viz_dir.mkdir(exist_ok=True)
            
            if metadata.get('task_type') in ['classification', 'regression']:
                viz_paths.extend(await self._generate_tabular_visualizations(
                    X, y, dataset_viz_dir, metadata.get('task_type')
                ))
            else:
                viz_paths.extend(await self._generate_image_visualizations(
                    X, y, dataset_viz_dir
                ))
            
            return viz_paths
            
        except Exception as e:
            logger.error(f"Error generating dataset visualizations: {str(e)}")
            return []
    
    async def _generate_tabular_visualizations(self, X: pd.DataFrame, y: pd.Series, 
                                             viz_dir: Path, task_type: str) -> List[str]:
        viz_paths = []
        
        # Data distribution plots
        viz_paths.append(await self._plot_data_distribution(X, viz_dir))
        
        # Target distribution
        viz_paths.append(await self._plot_target_distribution(y, viz_dir, task_type))
        
        # Correlation matrix
        if len(X.select_dtypes(include=[np.number]).columns) > 1:
            viz_paths.append(await self._plot_correlation_matrix(X, viz_dir))
        
        # Feature importance (if possible)
        viz_paths.append(await self._plot_feature_statistics(X, viz_dir))
        
        # Missing values heatmap
        if X.isnull().sum().sum() > 0:
            viz_paths.append(await self._plot_missing_values(X, viz_dir))
        
        return [p for p in viz_paths if p]
    
    async def _generate_image_visualizations(self, image_paths: List[str], 
                                           labels: List[str], viz_dir: Path) -> List[str]:
        viz_paths = []
        
        # Class distribution
        viz_paths.append(await self._plot_class_distribution(labels, viz_dir))
        
        # Sample images grid
        viz_paths.append(await self._plot_sample_images(image_paths, labels, viz_dir))
        
        return [p for p in viz_paths if p]
    
    async def generate_training_visualizations(self, task_id: str, model_type: str, 
                                             metrics_history: Dict[str, List], 
                                             task_type: str) -> List[str]:
        viz_paths = []
        user_viz_dir = self.viz_dir / "training" / task_id
        user_viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Training/Validation Loss
        if 'loss' in metrics_history:
            viz_paths.append(await self._plot_training_loss(metrics_history, user_viz_dir))
        
        # Accuracy/Performance Metrics
        if task_type == 'classification' and 'accuracy' in metrics_history:
            viz_paths.append(await self._plot_accuracy_curves(metrics_history, user_viz_dir))
        elif task_type == 'regression' and any(k in metrics_history for k in ['mse', 'mae', 'r2']):
            viz_paths.append(await self._plot_regression_metrics(metrics_history, user_viz_dir))
        
        # Learning rate schedule
        if 'learning_rate' in metrics_history:
            viz_paths.append(await self._plot_learning_rate(metrics_history, user_viz_dir))
        
        return [p for p in viz_paths if p]
    
    async def generate_evaluation_visualizations(self, y_true, y_pred, model_name: str, 
                                               user_name: str, task_type: str, 
                                               class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        user_viz_dir = Config.get_user_dir(self.viz_dir, user_name)
        model_viz_dir = user_viz_dir / f"evaluation_{model_name}"
        model_viz_dir.mkdir(exist_ok=True)
        
        viz_paths = []
        metrics = {}
        
        if task_type == 'classification':
            # Confusion Matrix
            viz_paths.append(await self._plot_confusion_matrix(
                y_true, y_pred, model_viz_dir, class_names
            ))
            
            # Classification Report
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            viz_paths.append(await self._plot_classification_report(report, model_viz_dir))
            
            # ROC Curves
            if len(np.unique(y_true)) == 2:
                viz_paths.append(await self._plot_roc_curve_binary(y_true, y_pred, model_viz_dir))
            else:
                viz_paths.append(await self._plot_roc_curve_multiclass(
                    y_true, y_pred, model_viz_dir, class_names
                ))
            
            metrics = {
                'accuracy': float(np.mean(y_true == y_pred)),
                'classification_report': report
            }
            
        elif task_type == 'regression':
            # Residual plots
            viz_paths.append(await self._plot_residuals(y_true, y_pred, model_viz_dir))
            
            # Prediction vs Actual
            viz_paths.append(await self._plot_prediction_vs_actual(y_true, y_pred, model_viz_dir))
            
            # Error distribution
            viz_paths.append(await self._plot_error_distribution(y_true, y_pred, model_viz_dir))
            
            metrics = {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred)))
            }
        
        return {
            'visualizations': [p for p in viz_paths if p],
            'metrics': metrics
        }
    
    async def _plot_data_distribution(self, X: pd.DataFrame, viz_dir: Path) -> Optional[str]:
        try:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return None
            
            n_cols = min(4, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:12]):
                if i < len(axes):
                    axes[i].hist(X[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            for i in range(len(numeric_cols), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            viz_path = viz_dir / 'data_distribution.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting data distribution: {str(e)}")
            return None
    
    async def _plot_target_distribution(self, y: pd.Series, viz_dir: Path, task_type: str) -> Optional[str]:
        try:
            plt.figure(figsize=self.figure_size)
            
            if task_type == 'classification':
                y.value_counts().plot(kind='bar')
                plt.title('Target Class Distribution')
                plt.xlabel('Classes')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
            else:
                plt.hist(y.dropna(), bins=50, alpha=0.7, edgecolor='black')
                plt.title('Target Variable Distribution')
                plt.xlabel('Target Value')
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            viz_path = viz_dir / 'target_distribution.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting target distribution: {str(e)}")
            return None
    
    async def _plot_correlation_matrix(self, X: pd.DataFrame, viz_dir: Path) -> Optional[str]:
        try:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return None
            
            corr_matrix = X[numeric_cols].corr()
            
            plt.figure(figsize=(min(len(numeric_cols), 15), min(len(numeric_cols), 15)))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            
            viz_path = viz_dir / 'correlation_matrix.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting correlation matrix: {str(e)}")
            return None
    
    async def _plot_confusion_matrix(self, y_true, y_pred, viz_dir: Path, 
                                   class_names: Optional[List[str]] = None) -> Optional[str]:
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names or range(len(cm)),
                       yticklabels=class_names or range(len(cm)))
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            
            viz_path = viz_dir / 'confusion_matrix.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            return None
    
    async def _plot_training_loss(self, metrics_history: Dict[str, List], viz_dir: Path) -> Optional[str]:
        try:
            plt.figure(figsize=self.figure_size)
            
            epochs = range(1, len(metrics_history['loss']) + 1)
            plt.plot(epochs, metrics_history['loss'], 'b-', label='Training Loss')
            
            if 'val_loss' in metrics_history:
                plt.plot(epochs, metrics_history['val_loss'], 'r-', label='Validation Loss')
            
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            viz_path = viz_dir / 'training_loss.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting training loss: {str(e)}")
            return None
    
    async def _plot_residuals(self, y_true, y_pred, viz_dir: Path) -> Optional[str]:
        try:
            residuals = y_true - y_pred
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Residuals vs Predicted
            ax1.scatter(y_pred, residuals, alpha=0.6)
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_xlabel('Predicted Values')
            ax1.set_ylabel('Residuals')
            ax1.set_title('Residuals vs Predicted')
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot of Residuals')
            
            plt.tight_layout()
            viz_path = viz_dir / 'residuals_analysis.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting residuals: {str(e)}")
            return None
    
    async def _plot_prediction_vs_actual(self, y_true, y_pred, viz_dir: Path) -> Optional[str]:
        try:
            plt.figure(figsize=self.figure_size)
            
            plt.scatter(y_true, y_pred, alpha=0.6)
            
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Predicted vs Actual Values')
            plt.grid(True, alpha=0.3)
            
            r2 = r2_score(y_true, y_pred)
            plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            viz_path = viz_dir / 'prediction_vs_actual.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting prediction vs actual: {str(e)}")
            return None
    
    async def _plot_feature_statistics(self, X: pd.DataFrame, viz_dir: Path) -> Optional[str]:
        try:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return None
            
            stats_df = X[numeric_cols].describe().T
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Mean values
            stats_df['mean'].plot(kind='bar', ax=axes[0,0], title='Mean Values')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Standard deviation
            stats_df['std'].plot(kind='bar', ax=axes[0,1], title='Standard Deviation')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Min-Max range
            axes[1,0].bar(range(len(stats_df)), stats_df['max'] - stats_df['min'])
            axes[1,0].set_title('Value Range (Max - Min)')
            axes[1,0].set_xticks(range(len(stats_df)))
            axes[1,0].set_xticklabels(stats_df.index, rotation=45)
            
            # Quartile ranges
            axes[1,1].bar(range(len(stats_df)), stats_df['75%'] - stats_df['25%'])
            axes[1,1].set_title('Interquartile Range')
            axes[1,1].set_xticks(range(len(stats_df)))
            axes[1,1].set_xticklabels(stats_df.index, rotation=45)
            
            plt.tight_layout()
            viz_path = viz_dir / 'feature_statistics.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting feature statistics: {str(e)}")
            return None
    
    async def _plot_missing_values(self, X: pd.DataFrame, viz_dir: Path) -> Optional[str]:
        try:
            missing_data = X.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) == 0:
                return None
            
            plt.figure(figsize=(12, 6))
            missing_data.plot(kind='bar')
            plt.title('Missing Values by Feature')
            plt.xlabel('Features')
            plt.ylabel('Number of Missing Values')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            viz_path = viz_dir / 'missing_values.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting missing values: {str(e)}")
            return None
    
    async def _plot_class_distribution(self, labels: List[str], viz_dir: Path) -> Optional[str]:
        try:
            label_counts = pd.Series(labels).value_counts()
            
            plt.figure(figsize=self.figure_size)
            label_counts.plot(kind='bar')
            plt.title('Class Distribution in Image Dataset')
            plt.xlabel('Classes')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            viz_path = viz_dir / 'class_distribution.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting class distribution: {str(e)}")
            return None
    
    async def _plot_sample_images(self, image_paths: List[str], labels: List[str], viz_dir: Path) -> Optional[str]:
        try:
            from PIL import Image
            import random
            
            unique_labels = list(set(labels))
            samples_per_class = min(3, len(image_paths) // len(unique_labels))
            
            fig, axes = plt.subplots(len(unique_labels), samples_per_class, 
                                   figsize=(samples_per_class*3, len(unique_labels)*3))
            
            if len(unique_labels) == 1:
                axes = [axes]
            
            for i, label in enumerate(unique_labels):
                label_images = [path for path, lbl in zip(image_paths, labels) if lbl == label]
                sample_images = random.sample(label_images, min(samples_per_class, len(label_images)))
                
                for j, img_path in enumerate(sample_images):
                    try:
                        img = Image.open(img_path)
                        if samples_per_class == 1:
                            ax = axes[i] if len(unique_labels) > 1 else axes[0]
                        else:
                            ax = axes[i][j] if len(unique_labels) > 1 else axes[j]
                        
                        ax.imshow(img)
                        ax.set_title(f'{label}')
                        ax.axis('off')
                    except Exception as e:
                        logger.warning(f"Could not load image {img_path}: {str(e)}")
            
            plt.tight_layout()
            viz_path = viz_dir / 'sample_images.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting sample images: {str(e)}")
            return None
    
    async def _plot_accuracy_curves(self, metrics_history: Dict[str, List], viz_dir: Path) -> Optional[str]:
        try:
            plt.figure(figsize=self.figure_size)
            
            epochs = range(1, len(metrics_history['accuracy']) + 1)
            plt.plot(epochs, metrics_history['accuracy'], 'b-', label='Training Accuracy')
            
            if 'val_accuracy' in metrics_history:
                plt.plot(epochs, metrics_history['val_accuracy'], 'r-', label='Validation Accuracy')
            
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            viz_path = viz_dir / 'accuracy_curves.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting accuracy curves: {str(e)}")
            return None
    
    async def _plot_regression_metrics(self, metrics_history: Dict[str, List], viz_dir: Path) -> Optional[str]:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            epochs = range(1, len(list(metrics_history.values())[0]) + 1)
            
            metrics_to_plot = ['mse', 'mae', 'r2']
            plot_idx = 0
            
            for metric in metrics_to_plot:
                if metric in metrics_history:
                    row, col = plot_idx // 2, plot_idx % 2
                    axes[row, col].plot(epochs, metrics_history[metric], 'b-', label=f'Training {metric.upper()}')
                    
                    val_metric = f'val_{metric}'
                    if val_metric in metrics_history:
                        axes[row, col].plot(epochs, metrics_history[val_metric], 'r-', label=f'Validation {metric.upper()}')
                    
                    axes[row, col].set_title(f'{metric.upper()} Over Time')
                    axes[row, col].set_xlabel('Epochs')
                    axes[row, col].set_ylabel(metric.upper())
                    axes[row, col].legend()
                    axes[row, col].grid(True, alpha=0.3)
                    plot_idx += 1
            
            # Hide unused subplot
            if plot_idx < 4:
                axes[1, 1].set_visible(False)
            
            plt.tight_layout()
            viz_path = viz_dir / 'regression_metrics.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting regression metrics: {str(e)}")
            return None
    
    async def _plot_learning_rate(self, metrics_history: Dict[str, List], viz_dir: Path) -> Optional[str]:
        try:
            plt.figure(figsize=self.figure_size)
            
            epochs = range(1, len(metrics_history['learning_rate']) + 1)
            plt.plot(epochs, metrics_history['learning_rate'], 'g-', linewidth=2)
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epochs')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            viz_path = viz_dir / 'learning_rate.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting learning rate: {str(e)}")
            return None
    
    async def _plot_classification_report(self, report: Dict, viz_dir: Path) -> Optional[str]:
        try:
            # Extract metrics for each class
            classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            metrics = ['precision', 'recall', 'f1-score']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(classes))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = [report[cls][metric] for cls in classes]
                ax.bar(x + i*width, values, width, label=metric.capitalize())
            
            ax.set_xlabel('Classes')
            ax.set_ylabel('Score')
            ax.set_title('Classification Report')
            ax.set_xticks(x + width)
            ax.set_xticklabels(classes, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            viz_path = viz_dir / 'classification_report.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting classification report: {str(e)}")
            return None
    
    async def _plot_roc_curve_binary(self, y_true, y_pred_proba, viz_dir: Path) -> Optional[str]:
        try:
            if hasattr(y_pred_proba, 'shape') and len(y_pred_proba.shape) > 1:
                y_pred_proba = y_pred_proba[:, 1]
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=self.figure_size)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            viz_path = viz_dir / 'roc_curve.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {str(e)}")
            return None
    
    async def _plot_roc_curve_multiclass(self, y_true, y_pred_proba, viz_dir: Path, 
                                       class_names: Optional[List[str]] = None) -> Optional[str]:
        try:
            n_classes = len(np.unique(y_true))
            
            # Binarize the output
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            plt.figure(figsize=self.figure_size)
            
            for i in range(n_classes):
                if hasattr(y_pred_proba, 'shape') and len(y_pred_proba.shape) > 1:
                    y_score = y_pred_proba[:, i]
                else:
                    continue
                    
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score)
                roc_auc = auc(fpr, tpr)
                
                class_name = class_names[i] if class_names else f'Class {i}'
                plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multi-class ROC Curves')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            viz_path = viz_dir / 'roc_curves_multiclass.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting multiclass ROC curves: {str(e)}")
            return None
    
    async def _plot_error_distribution(self, y_true, y_pred, viz_dir: Path) -> Optional[str]:
        try:
            errors = y_true - y_pred
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Error histogram
            ax1.hist(errors, bins=50, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Prediction Errors')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Error Distribution')
            ax1.axvline(x=0, color='r', linestyle='--', alpha=0.7)
            
            # Error statistics
            error_stats = {
                'Mean Error': np.mean(errors),
                'Std Error': np.std(errors),
                'MAE': np.mean(np.abs(errors)),
                'RMSE': np.sqrt(np.mean(errors**2))
            }
            
            ax2.bar(error_stats.keys(), error_stats.values())
            ax2.set_title('Error Statistics')
            ax2.set_ylabel('Value')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
            
            plt.tight_layout()
            viz_path = viz_dir / 'error_distribution.png'
            plt.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return str(viz_path.relative_to(self.viz_dir))
            
        except Exception as e:
            logger.error(f"Error plotting error distribution: {str(e)}")
            return None
    
    def save_metrics_json(self, metrics: Dict[str, Any], file_path: Path):
        try:
            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metrics JSON: {str(e)}")
    
    def create_interactive_plot(self, data: Dict[str, Any], plot_type: str, viz_dir: Path) -> Optional[str]:
        try:
            if plot_type == 'training_metrics':
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Loss', 'Accuracy', 'Learning Rate', 'Validation Metrics')
                )
                
                if 'loss' in data:
                    fig.add_trace(
                        go.Scatter(x=list(range(len(data['loss']))), y=data['loss'], 
                                 mode='lines', name='Training Loss'),
                        row=1, col=1
                    )
                
                if 'accuracy' in data:
                    fig.add_trace(
                        go.Scatter(x=list(range(len(data['accuracy']))), y=data['accuracy'],
                                 mode='lines', name='Training Accuracy'),
                        row=1, col=2
                    )
                
                fig.update_layout(height=600, showlegend=True, title_text="Training Metrics Dashboard")
                
                viz_path = viz_dir / 'interactive_training_metrics.html'
                fig.write_html(str(viz_path))
                
                return str(viz_path.relative_to(self.viz_dir))
                
        except Exception as e:
            logger.error(f"Error creating interactive plot: {str(e)}")
            return None