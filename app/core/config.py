import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model artifact paths
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
PCA_PATH = os.path.join(MODELS_DIR, "pca.pkl")
KNN_PATH = os.path.join(MODELS_DIR, "knn_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Ollama / NLP Settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
NLP_NER_MODEL = os.getenv("NLP_NER_MODEL", "symptemist-onnx-int8")
