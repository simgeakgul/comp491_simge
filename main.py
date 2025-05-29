from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
import os
import zipfile
import uuid
import sys
import run_generate_prompt
import run_others
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = "jobs"
os.makedirs(JOBS_DIR, exist_ok=True)

# 1. Generate prompts
@app.post("/generate_prompt")
async def generate_prompt(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fovdeg: float = Form(...),
    fovmap_atmosphere: float = Form(...),
    fovmap_sky_or_ceiling: float = Form(...),
    fovmap_ground_or_floor: float = Form(...),
    in_out: str = Form(...),
    guidance_scale: float = Form(...),
    key: str = Form(...),
    name: str = Form(...)
):
    if key != "living-paintings":
        raise HTTPException(status_code=403, detail="Invalid API key")

    config =\
    {
        "out_w": 4096,
        "out_h": 2048,
        "pitch_map": {
          "atmosphere": 0.0,
          "sky_or_ceiling": 45.0,
          "ground_or_floor": -45.0
        },
        "horizontal_yaws": [45, -45, 135, -135, 90, -90],
        "sky_yaws": [0, 90, 180, 270],
        "ground_yaws": [0, 90, 180, 270],
        "fov_map": {
          "atmosphere": 80.0,
          "sky_or_ceiling": 100.0,
          "ground_or_floor": 100.0
        },
        "guidance_scale": 7.0,
        "steps": 50,
        "dilate_pixel": 16,
        "fovdeg": 95.0,
        "border_px": 15,
        "crop_size": 1024,
        "edge_sigma": 3.0,
        "center_bias": 1.0,
        "align_depth": False,
        "in_out": "outdoor",
        "name": name
    }

    # Update config with form data
    config["fovdeg"] = fovdeg
    config["fov_map"]["atmosphere"] = fovmap_atmosphere
    config["fov_map"]["sky_or_ceiling"] = fovmap_sky_or_ceiling
    config["fov_map"]["ground_or_floor"] = fovmap_ground_or_floor
    config["in_out"] = in_out
    config["guidance_scale"] = guidance_scale

    # Create job directory
    job_id = str(uuid.uuid4())
    job_path = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_path, exist_ok=True)

    # Save input files
    input_image_path = os.path.join(job_path, "input.jpg")
    with open(input_image_path, "wb") as f:
        f.write(await file.read())

    config_path = os.path.join(job_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Initialize prompt status
    with open(os.path.join(job_path, "prompt_status.txt"), "w") as f:
        f.write("processing")

    with open(os.path.join(job_path, "status.txt"), "w") as f:
        f.write("generating prompts")

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
    atmosphere_prompt: str = Form(...),
    sky_or_ceiling_prompt: str = Form(...),
    ground_or_floor_prompt: str = Form(...),
):
    job_path = os.path.join(JOBS_DIR, job_id)
    if not os.path.exists(job_path):
        raise HTTPException(status_code=404, detail="Invalid job_id")

    # Save edited prompts as JSON
    prompts = {
        "atmosphere": atmosphere_prompt,
        "sky_or_ceiling": sky_or_ceiling_prompt,
        "ground_or_floor": ground_or_floor_prompt
    }

    edited_prompts_path = os.path.join(job_path, "prompts.json")
    with open(edited_prompts_path, "w") as f:
        json.dump(prompts, f, indent=2)

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

# 1) Fixed panorama
@app.get("/pano/{job_id}")
def get_pano(job_id: str):
    job_path = os.path.join(JOBS_DIR, job_id)
    fixed_pano = os.path.join(job_path, "fixed_pano.jpg")
    if not os.path.exists(fixed_pano):
        raise HTTPException(status_code=404, detail="Fixed panorama not ready")
    return FileResponse(
        fixed_pano,
        media_type="image/jpeg",
        filename="fixed_pano.jpg"
    )

# 2) Depth panorama
@app.get("/depth/{job_id}")
def get_depth(job_id: str):
    job_path = os.path.join(JOBS_DIR, job_id)
    depth_pano = os.path.join(job_path, "depth_pano.jpg")
    if not os.path.exists(depth_pano):
        raise HTTPException(status_code=404, detail="Depth panorama not ready")
    return FileResponse(
        depth_pano,
        media_type="image/jpeg",
        filename="depth_pano.jpg"
    )

# 3) Soundscape
@app.get("/soundscape/{job_id}")
def get_soundscape(job_id: str):
    job_path = os.path.join(JOBS_DIR, job_id)
    soundscape = os.path.join(job_path, "soundscape.wav")
    if not os.path.exists(soundscape):
        raise HTTPException(status_code=404, detail="Soundscape not ready")
    return FileResponse(
        soundscape,
        media_type="audio/wav",
        filename="soundscape.wav"
    )

@app.get("/all_meta")
def get_all_meta():
    result = {}
    for job_id in os.listdir(JOBS_DIR):
        job_path = os.path.join(JOBS_DIR, job_id)
        if not os.path.isdir(job_path):
            continue

        status_file = os.path.join(job_path, "status.txt")
        config_file = os.path.join(job_path, "config.json")
        if os.path.exists(status_file):
            with open(status_file) as f:
                status = f.read().strip()
        else:
            status = "unknown"

        result[job_id] = {}
        result[job_id]["status"] = status
        if os.path.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)
            result[job_id]["environment"] = config["in_out"]
            result[job_id]["name"] = config["name"]


    if not result:
        raise HTTPException(status_code=404, detail="No jobs found")

    return result



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
            f.write("waiting prompts")

    except Exception as e:
        with open(os.path.join(job_path, "prompt_status.txt"), "w") as f:
            f.write(f"error: {str(e)}")