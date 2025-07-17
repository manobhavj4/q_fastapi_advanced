from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routers import sensing, compute, jobs

from global_services.credential_manager import GLOBAL_CREDENTIALS
from global_services.global_services import GlobalServices
from contextlib import asynccontextmanager
from edge_sensor import edge_main
from ai_control_engine import ai_control_main
from quantum_control import quantum_control_main
import asyncio
from global_services.get_global_context import init_global_services


# global_context.py
gs = None
logger = None
env_config = {}

@asynccontextmanager
async def lifespan(app: FastAPI):

    gs, logger, env_config = init_global_services()
    
    logger.info("✅ App started.")
    logger.info(f"AWS Region: {env_config.get('AWS_REGION')}")

    yield

    # Shutdown (optional)
    logger.info("🛑 App shutting down...")

    # Gracefully close log handlers
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
    await asyncio.sleep(0.1)  # tiny delay to flush logs
    

#fastapi call app
app = FastAPI(lifespan=lifespan,
    title="QuSP API",
    description="Backend APIs for Quantum Sensing Platform",
    version="1.0.0"
)

# CORS settings – adjust for your frontend domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if not GLOBAL_CREDENTIALS:
    gs = GlobalServices()
    logger = gs.get_logger()
    env_config=gs.get_credential()


logger.info(f"aws region: {env_config['AWS_REGION']}")

# Refresh from AWS using all keys with values from env and aws

logger.info(f"Postgres User: {env_config['POSTGRES_USER']}")
logger.info(f"Mongo URI: {env_config['MONGO_URI']}")


# Root route
@app.get("/")
async def root():
    return {"message": "Welcome to the QuSP API!"}




# Include all API routes
app.include_router(sensing.router, prefix="/api/sensing", tags=["Sensors sensing"])
app.include_router(compute.router, prefix="/api/compute", tags=["Quantum Compute"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["Job Manager"])

app.include_router(edge_main.router, prefix="/api/edgesensor", tags=["Edge Sensor"])

app.include_router(ai_control_main.router, prefix="/api/aicontrol", tags=["AI Control Engine"])

app.include_router(quantum_control_main.router, prefix="/api/quantumcontrol", tags=["Quantum Control Engine"])
