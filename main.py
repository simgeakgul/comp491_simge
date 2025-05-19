from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
import os
import zipfile
import uuid
import sys
import run_generate_prompt
import run_others
import json

app = FastAPI()

JOBS_DIR = "jobs"
os.makedirs(JOBS_DIR, exist_ok=True)

# 1. Generate prompts
@app.post("/generate_prompt")
async def generate_prompt(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config: UploadFile = File(...)
):
    # Create job directory
    job_id = str(uuid.uuid4())
    job_path = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_path, exist_ok=True)

    # Save input files
    input_image_path = os.path.join(job_path, file.filename)
    with open(input_image_path, "wb") as f:
        f.write(await file.read())

    config_path = os.path.join(job_path, "config.yaml")
    with open(config_path, "wb") as f:
        f.write(await config.read())

    # Initialize prompt status
    with open(os.path.join(job_path, "prompt_status.txt"), "w") as f:
        f.write("processing")

    # Run in background
    background_tasks.add_task(run_prompt_generation_pipeline, job_path)

    return {"job_id": job_id}


@app.get("/prompt_status/{job_id}")
def get_prompt_status(job_id: str):
    job_path = os.path.join(JOBS_DIR, job_id)
    prompt_status_path = os.path.join(job_path, "prompt_status.txt")
    prompt_path = os.path.join(job_path, "prompts.json")

    if not os.path.exists(prompt_status_path):
        raise HTTPException(status_code=404, detail="Invalid job_id or prompt generation not started")

    with open(prompt_status_path) as f:
        status = f.read().strip()

    if status == "done" and os.path.exists(prompt_path):
        with open(prompt_path, "r") as pf:
            prompts = json.load(pf)  # ✅ parse JSON
        return {
            "job_id": job_id,
            "prompt_status": status,
            "prompts": prompts  # ✅ return actual JSON data
        }

    return {"job_id": job_id, "prompt_status": status}



# 2. Submit edited prompts and run the rest of the pipeline
@app.post("/run_with_prompts")
async def run_with_prompts(
    background_tasks: BackgroundTasks,
    job_id: str = Form(...),
    edited_prompts: UploadFile = File(...)
):
    job_path = os.path.join(JOBS_DIR, job_id)
    if not os.path.exists(job_path):
        raise HTTPException(status_code=404, detail="Invalid job_id")

    # Save edited prompts as JSON
    edited_prompts_path = os.path.join(job_path, "prompts.json")
    try:
        contents = await edited_prompts.read()
        parsed_json = json.loads(contents)  # ✅ Validate it's proper JSON
        with open(edited_prompts_path, "w") as f:
            json.dump(parsed_json, f, indent=2)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in edited_prompts")

    # Update status
    with open(os.path.join(job_path, "status.txt"), "w") as f:
        f.write("processing")

    background_tasks.add_task(run_others_pipeline, job_path)

    return {"job_id": job_id, "status": "processing"}


# 3. Poll job status
@app.get("/status/{job_id}")
def get_status(job_id: str):
    status_path = os.path.join(JOBS_DIR, job_id, "status.txt")
    if not os.path.exists(status_path):
        raise HTTPException(status_code=404, detail="Invalid job_id")
    with open(status_path) as f:
        status = f.read().strip()
    return {"job_id": job_id, "status": status}


# 4. Download results
@app.get("/results/{job_id}")
def get_results(job_id: str):
    zip_path = os.path.join(JOBS_DIR, job_id, "results.zip")
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Results not ready")
    return FileResponse(zip_path, media_type="application/zip", filename="results.zip")


# 🔁 Pipeline Step 2 (run_others.py)
def run_others_pipeline(job_path: str):
    try:
        sys.argv = ["run_others.py", "--base", job_path]
        run_others.main()

        fixed_pano = os.path.join(job_path, "fixed_pano.jpg")
        depth_pano = os.path.join(job_path, "depth_pano.jpg")
        soundscape = os.path.join(job_path, "soundscape.wav")
        if not (os.path.exists(fixed_pano) and os.path.exists(depth_pano)):
            raise Exception("Missing output images")

        zip_path = os.path.join(job_path, "results.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(fixed_pano, arcname="fixed_pano.jpg")
            zipf.write(depth_pano, arcname="depth_pano.jpg")
            zipf.write(soundscape, arcname="soundscape.wav")

        with open(os.path.join(job_path, "status.txt"), "w") as f:
            f.write("done")

    except Exception as e:
        with open(os.path.join(job_path, "status.txt"), "w") as f:
            f.write(f"error: {str(e)}")


def run_prompt_generation_pipeline(job_path: str):
    try:
        sys.argv = ["run_generate_prompt.py", "--base", job_path]
        run_generate_prompt.main()

        prompt_path = os.path.join(job_path, "prompts.json")
        if not os.path.exists(prompt_path):
            raise Exception("Prompt generation output not found")

        with open(os.path.join(job_path, "prompt_status.txt"), "w") as f:
            f.write("done")

        with open(os.path.join(job_path, "status.txt"), "w") as f:
            f.write("awaiting_prompt_edit")

    except Exception as e:
        with open(os.path.join(job_path, "prompt_status.txt"), "w") as f:
            f.write(f"error: {str(e)}")