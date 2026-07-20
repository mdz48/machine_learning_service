from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import engine, Base
from app.features.risk_prediction.routes.risk_router import router as risk_router
from app.features.nlp_symptom_extraction.routes.nlp_router import router as nlp_router

app = FastAPI(title="ML Service", root_path="/ml")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables at startup if missing
Base.metadata.create_all(bind=engine)

# Register Feature Routers (Vertical Slices)
app.include_router(risk_router)
app.include_router(nlp_router)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "ML Service is running"}
