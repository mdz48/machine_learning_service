import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import Base, MLModel, SessionLocal, engine

Base.metadata.create_all(bind=engine)

MODEL_DATA = {
    "version": "v2.0.0",
    "training_date": "2026-07-05",
    "algorithm": "PCA + K-Means (descubrimiento) + KNN (inferencia)",
    "metadata_config": {
        "pca_components": 5,
        "pca_variance_pct": 53.62,
        "kmeans_k": 4,
        "n_neighbors": 15,
        "weights": "uniform",
        "metric": "euclidean",
        "pipeline_steps": [
            "Missing Imputation (Median / Most-Frequent)",
            "StandardScaler",
            "OneHotEncoder (drop=first)",
            "PCA (5 components)",
            "KNN (k=15)"
        ],
        "author": "Maximiliano Diaz"
    }
}


def seed():
    db = SessionLocal()
    try:
        existing = db.query(MLModel).filter(MLModel.version == MODEL_DATA["version"]).first()
        if existing:
            print(f"El modelo {MODEL_DATA['version']} ya existe con id={existing.id}.")
            return

        db.query(MLModel).update({"is_active": False})

        new_model = MLModel(
            version=MODEL_DATA["version"],
            training_date=MODEL_DATA["training_date"],
            algorithm=MODEL_DATA["algorithm"],
            metadata_config=MODEL_DATA["metadata_config"],
            is_active=True
        )
        db.add(new_model)
        db.commit()
        db.refresh(new_model)
        print(f"Modelo registrado correctamente con id={new_model.id}, version={new_model.version}, is_active=True")
    finally:
        db.close()

if __name__ == "__main__":
    seed()
