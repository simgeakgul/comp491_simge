from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
import os
import zipfile
import uuid
import run_all
import sys
import yaml  # For writing config.yaml

app = FastAPI()

JOBS_DIR = "jobs"
os.makedirs(JOBS_DIR, exist_ok=True)

# Start background job with file upload and fov_map inputs
@app.post("/run")
async def run(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config: UploadFile = File(...)
):
    # Generate unique job ID and path
    job_id = str(uuid.uuid4())
    job_path = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_path, exist_ok=True)

    # Save uploaded image
    input_image_path = os.path.join(job_path, file.filename)
    with open(input_image_path, "wb") as f:
        f.write(await file.read())

    # Save uploaded config.yaml
    config_path = os.path.join(job_path, "config.yaml")
    with open(config_path, "wb") as f:
        f.write(await config.read())

    # Write initial status
    with open(os.path.join(job_path, "status.txt"), "w") as f:
        f.write("processing")

    # Start the pipeline
    background_tasks.add_task(run_pipeline, job_path, input_image_path)
    return {"job_id": job_id, "status": "processing"}

# Poll job status
@app.get("/status/{job_id}")
def get_status(job_id: str):
    status_path = os.path.join(JOBS_DIR, job_id, "status.txt")
    if not os.path.exists(status_path):
        raise HTTPException(status_code=404, detail="Invalid job_id")

    with open(status_path) as f:
        status = f.read().strip()

    return {"job_id": job_id, "status": status}

# Download result
@app.get("/results/{job_id}")
def get_results(job_id: str):
    zip_path = os.path.join(JOBS_DIR, job_id, "results.zip")
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Results not ready")

    return FileResponse(zip_path, media_type="application/zip", filename="results.zip")

# Pipeline runner
def run_pipeline(job_path: str, input_image_path: str):
    try:
        # Call run_all.py with new input base
        sys.argv = ["run_all.py", "--base", job_path]
        run_all.main()

        # Expected output files
        fixed_pano = os.path.join(job_path, "fixed_pano.jpg")
        depth_pano = os.path.join(job_path, "depth_pano.jpg")
        if not (os.path.exists(fixed_pano) and os.path.exists(depth_pano)):
            raise Exception("Missing output images")

        # Create zip
        zip_path = os.path.join(job_path, "results.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(fixed_pano, arcname="fixed_pano.jpg")
            zipf.write(depth_pano, arcname="depth_pano.jpg")

        # Mark as done
        with open(os.path.join(job_path, "status.txt"), "w") as f:
            f.write("done")

    except Exception as e:
        with open(os.path.join(job_path, "status.txt"), "w") as f:
            f.write(f"error: {str(e)}")
