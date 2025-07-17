from fastapi import APIRouter
from quantum_control.circuits.qaoa_example import run_qaoa

router = APIRouter()

# run_qaoa(depth: int, gamma: float)
# Call via /api/sensors/nipan
@router.get("/nipan")
async def read_specific_sensor():
    return run_qaoa(15,74.4)
    # return {"message": "Sensors endpoint working!"}

# Call via /api/sensors/
@router.get("/")
async def read_sensors():
    return {"message": "Sensors endpoint working!"}
