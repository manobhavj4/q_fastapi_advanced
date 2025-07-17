from global_services.credential_manager import load_credentials, GLOBAL_CREDENTIALS
from global_services.logger_factory import create_logger
from global_services.config_services import AWSConfigService  # Optional AWS helper

 
class GlobalServices:
    """
    Centralized service manager for environment configs, logger, and AWS access.
    """
    def __init__(self, user: str = ""):
        # Step 1: Load credentials from .env and after initialise we can use global variable GLOBAL_CREDENTIALS
        # if not GLOBAL_CREDENTIALS:
        self.credentials = load_credentials()

        # Step 2: Setup logger (system ID + date-based)
        self.logger = create_logger(user)
        self.logger.info("GlobalServices initialized.")

        # Step 3: Optional - setup AWS service access
        try:
            self.aws_config = AWSConfigService(region_name=self.credentials.get("AWS_REGION"))
        except Exception as e:
            self.logger.error(f"[AWS_PARAM] Failed to connect to AWS: {e}")
        
    def get_credential(self, fallback: str = None):
        aws_keys = list(self.credentials.keys())
        for key in aws_keys:
            try:
                value = self.aws_config.get_parameter(key, with_decryption=True)
                GLOBAL_CREDENTIALS[key] = value
                self.logger.info(f"Fetched and stored AWS param: {key}")

            except Exception as e:
                self.logger.error(f"[AWS_PARAM] Failed to fetch {key}: {e}")
        
        return GLOBAL_CREDENTIALS.copy()
        # return GLOBAL_CREDENTIALS.get(key, fallback)

    def get_aws_param(self, param_name: str, decrypt: bool = True):
        """
        Fetch from AWS SSM Parameter Store
        """
        try:
            return self.aws_config.get_parameter(param_name, with_decryption=decrypt)
        except Exception as e:
            self.logger.error(f"[AWS_PARAM] Failed to fetch {param_name}: {e}")
            return None
        

    def fetch_all_credentials(self, decrypt: bool = True):
        """
        Fetch list of AWS SSM keys and store them in global env dict.
        """
        aws_keys = list(self.credentials.keys())
        for key in aws_keys:
            try:
                value = self.aws_config.get_parameter(key, with_decryption=decrypt)
                GLOBAL_CREDENTIALS[key] = value
                self.logger.info(f"Fetched and stored AWS param: {key}")

            except Exception as e:
                self.logger.error(f"[AWS_PARAM] Failed to fetch {key}: {e}")
        
        return GLOBAL_CREDENTIALS.copy()

    def get_logger(self):
        return self.logger
