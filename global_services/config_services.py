# global_services/config_service.py
import boto3
from global_services.get_global_context import logger
from global_services.logger_factory import create_logger
if logger is None:
    logger = create_logger()



class AWSConfigService:
    def __init__(self, region_name="us-east-1"):
        self.ssm = boto3.client("ssm", region_name=region_name)

    def get_parameter(self, name: str, with_decryption: bool = True):
        try:
            response = self.ssm.get_parameter(Name=name, WithDecryption=with_decryption)
            return response["Parameter"]["Value"]
        except Exception as e:
            logger.info(f"[ERROR] Failed to fetch '{name}' from AWS SSM: {e}")
            return None
