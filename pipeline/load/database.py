from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from urllib.parse import quote_plus
from ..config import settings

DB_USER = settings.DB_USER.get_secret_value()
DB_PASSWORD_ENCODED = quote_plus(settings.DB_PASSWORD.get_secret_value()).replace('%', '%%')   # url encoding
DB_HOST = settings.DB_HOST.get_secret_value()
DB_PORT = settings.DB_PORT
DB_NAME = settings.DB_NAME.get_secret_value()

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(
    DB_URL,
    pool_pre_ping = True,
    pool_recycle = 1800,
)

Session = sessionmaker(
    autocommit = False, 
    autoflush = False, 
    bind = engine, 
    expire_on_commit = False)