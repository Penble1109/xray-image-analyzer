from datetime import datetime
from pathlib import Path
import json
import numpy as np
import cv2

def save_outputs( img: np.ndarray, flags: dict, brightness: int, contrast: int, outdir: str | None = None,) -> tuple[str, str]:
    out_dir = Path(outdir) if outdir else Path.cwd() / "outputs"
    out_dir.mkdir(parents=True, exist_ok = True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = out_dir / f"processed_{ts}.png"
    ok = cv2.imwrite(str(img_path),img)

    if not ok:
        raise RuntimeError("cv2.imwrite failed")
    
    report = {
        "timestamp": ts,
        "brightness":int(brightness),
        "contrast" : int(contrast),
        "flags": flags,
        "image_path": str(img_path),
    }

    json_path = out_dir / f"report_{ts}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(img_path), str(json_path)