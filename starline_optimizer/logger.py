import sys
from loguru import logger
from .env import APP_ENV

# Serialize logs as JSON in production
if APP_ENV == "production":
    logger.remove(0)
    logger.add(sys.stderr, format="{message}", serialize=True)
