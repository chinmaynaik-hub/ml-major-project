from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
from typing import List

router = APIRouter(prefix="/dataset", tags=["dataset"])

DATASET_ROOT = Path("yoga_ai_trainer/backend/data/raw/yoga_poses/dataset")

@router.post("/upload/{pose_name}")
async def upload_pose_images(pose_name: str, files: List[UploadFile] = File(...)):
    pose_dir = DATASET_ROOT / pose_name
    pose_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for file in files:
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
        target = pose_dir / file.filename
        with target.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        saved += 1
    return {"status": "success", "pose": pose_name, "saved": saved}

@router.get("/list")
async def list_dataset():
    if not DATASET_ROOT.exists():
        return {"poses": [], "total_images": 0}
    poses = []
    total_images = 0
    for pose_dir in DATASET_ROOT.iterdir():
        if pose_dir.is_dir():
            count = sum(1 for p in pose_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"])
            poses.append({"pose": pose_dir.name, "count": count})
            total_images += count
    return {"poses": poses, "total_images": total_images}
