from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_PATH: str = "models/best_model_v1.pth"
    MODEL_VERSION: str = "1.0.0"
    IMAGE_SIZE: int = 224
    ENV: str = "dev"

    class Config:
        env_file = ".env"


settings = Settings()