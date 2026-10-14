"""
Extrae landmarks (cuerpo, manos, cara) de videos de senas con MediaPipe Holistic.
Uso:
  python extract_landmarks.py videos/hola.mp4
  python extract_landmarks.py videos/        (procesa todos los videos de la carpeta)
  python extract_landmarks.py landmarks/     (solo rellena huecos de JSON ya generados)
Salida: landmarks/<nombre>.json  -> se carga en viewer.html
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def lm_list(lms, vis=False):
    if lms is None:
        return None
    out = []
    for p in lms.landmark:
        d = {"x": round(p.x, 5), "y": round(p.y, 5), "z": round(p.z, 5)}
        if vis:
            d["visibility"] = round(p.visibility, 4)
        out.append(d)
    return out


def fill_gaps(frames, key, max_gap):
    """Rellena huecos cortos de una mano interpolando entre el frame anterior y el siguiente."""
    idx = [i for i, f in enumerate(frames) if f[key]]
    filled = 0
    for a, b in zip(idx, idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            A, B = frames[a][key], frames[b][key]
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                frames[a + k][key] = [
                    {c: round(pa[c] + (pb[c] - pa[c]) * t, 5) for c in ("x", "y", "z")}
                    for pa, pb in zip(A, B)
                ]
                filled += 1
    return filled


def report(frames, fps):
    """Cobertura de cada mano solo en frames donde la muneca esta dentro del cuadro."""
    max_gap = max(3, int(fps * 0.35))
    out = []
    for key, wrist, name in (("leftHand", 15, "izq"), ("rightHand", 16, "der")):
        filled = fill_gaps(frames, key, max_gap)
        on = [f for f in frames if f["pose"] and f["pose"][wrist]["y"] < 0.95
              and f["pose"][wrist]["visibility"] > 0.5]
        if not on:
            out.append(f"mano {name}: fuera de cuadro (no se usa)")
            continue
        cov = sum(1 for f in on if f[key]) / len(on)
        out.append(f"mano {name}: {100*cov:.0f}% ({filled} frames rellenados)")
        if cov < 0.8 and len(on) > 0.3 * len(frames):
            out.append(f"[!] mano {name} se pierde mucho estando en cuadro: mas luz (menos blur) o mas lento")
    return out


def save(data, out):
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))


def process(video: Path, out_dir: Path, complexity: int, face: bool):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[X] No se pudo abrir {video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames, n_l, n_r, n_pose = [], 0, 0, 0
    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=complexity,
        smooth_landmarks=True,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hol:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            r = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            f = {
                "pose": lm_list(r.pose_landmarks, True),
                "poseWorld": lm_list(r.pose_world_landmarks, True),
                "leftHand": lm_list(r.left_hand_landmarks),
                "rightHand": lm_list(r.right_hand_landmarks),
                "face": lm_list(r.face_landmarks) if face else None,
            }
            n_pose += f["pose"] is not None
            n_l += f["leftHand"] is not None
            n_r += f["rightHand"] is not None
            frames.append(f)
            if len(frames) % 30 == 0:
                print(f"  {video.name}: {len(frames)} frames...", end="\r")
    cap.release()

    data = {"name": video.stem, "fps": fps, "width": w, "height": h, "frames": frames}
    lines = report(frames, fps)
    out = out_dir / f"{video.stem}.json"
    save(data, out)
    print(f"[OK] {video.name} -> {out}  ({len(frames)} frames, {fps:.0f} fps)")
    for l in lines:
        print("     " + l)


def refix(path: Path):
    """Aplica relleno de huecos a un JSON ya generado, sin reprocesar el video."""
    data = json.load(open(path, encoding="utf-8"))
    lines = report(data["frames"], data["fps"])
    save(data, path)
    print(f"[OK] {path.name} corregido")
    for l in lines:
        print("     " + l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="video o carpeta de videos")
    ap.add_argument("-o", "--out", default="landmarks", help="carpeta de salida")
    ap.add_argument("--complexity", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--no-face", action="store_true", help="no guardar cara (JSON mas liviano)")
    a = ap.parse_args()

    src = Path(a.input)
    if src.suffix.lower() == ".json":
        return refix(src)
    if src.is_dir() and not any(p.suffix.lower() in VIDEO_EXT for p in src.iterdir()):
        for j in sorted(src.glob("*.json")):
            refix(j)
        return
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXT) if src.is_dir() else [src]
    if not videos:
        sys.exit("No hay videos para procesar.")
    for v in videos:
        process(v, out_dir, a.complexity, not a.no_face)


if __name__ == "__main__":
    main()
