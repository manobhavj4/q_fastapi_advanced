# global_context.py

gs = None
logger = None
env_config = {}

def init_global_services():
    global gs, logger, env_config
    if gs is None:
        from global_services.global_services import GlobalServices
        gs = GlobalServices()
        logger = gs.get_logger()
        env_config = gs.get_credential()
    return gs, logger, env_config
