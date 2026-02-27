from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
import httpx
from typing import List, Optional, Literal

class Settings(BaseSettings):
    DB_USER: SecretStr
    DB_PASSWORD: SecretStr
    DB_HOST: SecretStr
    DB_PORT: int
    DB_NAME: SecretStr

    model_config = SettingsConfigDict(env_file = '.env', env_file_encoding = 'utf-8',
                                      extra = 'ignore')

settings = Settings()
