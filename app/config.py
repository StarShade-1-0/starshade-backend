import os
from pydantic import BaseModel

class Settings(BaseModel):
    ALLOW_ORIGINS: str = os.getenv("ALLOW_ORIGINS", "*")  # comma-separated for prod
    ENV: str = os.getenv("ENV", "dev")

settings = Settings()
