import os

__REQUIRED_ENV_VARS = [
    "APP_ENV",
]

for key in __REQUIRED_ENV_VARS:
    if key not in os.environ:
        raise OSError(f"Required environment variable {key} is missing.")

APP_ENV = os.getenv("APP_ENV", default="development")
