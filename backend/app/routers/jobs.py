from fastapi import APIRouter

router = APIRouter()

#path is /api/jobs
@router.get("/")
async def get_job_status():
    return {"message": "Job Manager endpoint is working!"}


@router.get("/testq")
async def get_job_status():
    return {"message": "Job Manager endpoint is working!"}
