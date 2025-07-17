# global_services/defaults_config.py

DEFAULT_ENV_VARS = {
    "AWS_REGION": "us-west-2",
    "S3_BUCKET": "default-bucket",
    "DB_ENDPOINT": "sqlite:///:memory:",
    "LOG_LEVEL": "INFO",
    "MODEL_PATH": "./models/",
    "API_KEY": None,
    "SECRET_TOKEN": None,
    "POSTGRES_USER":"manobhav",
    "MONGO_URI":"https://mongodb"
}
