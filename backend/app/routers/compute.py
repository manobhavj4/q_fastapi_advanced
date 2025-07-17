# backend/app/routers/compute.py

from fastapi import APIRouter

router = APIRouter()

# path as /api/compute
@router.get("/")
async def run_quantum_job():
    return {"message": "Quantum compute endpoint is working!"}
