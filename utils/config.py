from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent.parent
    UPLOAD_DIR = BASE_DIR / "uploads"
    MODELS_DIR = BASE_DIR / "trained_models"
    TEMP_DIR = BASE_DIR / "temp"
    VIZ_DIR = BASE_DIR / "visualizations"
    STATIC_DIR = BASE_DIR / "static"
    LOGS_DIR = BASE_DIR / "logs"
    
    MODEL_EXTENSIONS = {
        "pytorch": ".pth",
        "sklearn": ".pkl",
        "yolo": ".pt",
        "tensorflow": ".h5"
    }
    
    SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    SUPPORTED_TABULAR_FORMATS = ['.csv', '.xlsx', '.xls']
    
    MAX_FILE_SIZE = 500 * 1024 * 1024
    MAX_TRAINING_TIME = 3600
    
    DEFAULT_TRAIN_PARAMS = {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "validation_split": 0.2,
        "early_stopping_patience": 10
    }
    
    VISUALIZATION_SETTINGS = {
        "figure_size": (12, 8),
        "dpi": 300,
        "format": "png",
        "style": "seaborn-v0_8"
    }
    
    @classmethod
    def setup_directories(cls):
        for dir_path in [cls.UPLOAD_DIR, cls.MODELS_DIR, cls.TEMP_DIR, 
                        cls.VIZ_DIR, cls.STATIC_DIR, cls.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_user_dir(cls, base_dir: Path, user_name: str) -> Path:
        user_dir = base_dir / user_name
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir