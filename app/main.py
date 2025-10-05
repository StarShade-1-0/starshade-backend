from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from matplotlib.pylab import f
from .config import settings
from .routers.k2 import router as k2_router
from .routers.kepler import router as kepler_router

app = FastAPI(title="Exoplanet Prediction API", version="1.0.0")

# CORS
origins = [o.strip() for o in settings.ALLOW_ORIGINS.split(",")] if settings.ALLOW_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "env": settings.ENV}

# Routers
app.include_router(k2_router)
app.include_router(kepler_router)
