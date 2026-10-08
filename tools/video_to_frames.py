r"""
video_to_frames.py
Turns videos into images for labelling (Roboflow / CVAT).

Usage (Windows cmd):
    python video_to_frames.py <video file or folder> [options]

Examples:
    python video_to_frames.py C:\yolo\videos
    python video_to_frames.py C:\yolo\videos --fps 5 --out C:\yolo\frames
    python video_to_frames.py C:\yolo\videos\IMG_0169.MOV --fps 8
    python video_to_frames.py C:\yolo\videos --blur 20     (also skip very blurry frames)

Install once:
    pip install opencv-python
"""

import argparse
import os
import sys

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".webm"}


def find_videos(path):
    """Return a list of video files: the file itself, or all videos in a folder."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS
        )
    return []


def sharpness(image):
    """Higher = sharper. Uses the variance of the Laplacian."""
    # Score on a fixed 640px-wide copy so 1080p and 4K videos are judged the same way
    h, w = image.shape[:2]
    small = cv2.resize(image, (640, max(1, int(h * 640 / w)))) if w > 640 else image
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract(video_path, out_root, fps, blur_limit):
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  !! Could not open {video_path} (see the HEVC note in the instructions)")
        return 0, 0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(video_fps / fps))  # e.g. 30 fps video, 5 fps wanted -> every 6th frame

    index = saved = skipped = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            if blur_limit > 0 and sharpness(frame) < blur_limit:
                skipped += 1
            else:
                saved += 1
                cv2.imwrite(os.path.join(out_dir, f"{name}_{saved:04d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
        index += 1

    cap.release()
    return saved, skipped


def main():
    parser = argparse.ArgumentParser(description="Extract images from videos for labelling.")
    parser.add_argument("input", help="A video file or a folder of videos")
    parser.add_argument("--out", default="frames", help="Output folder (default: frames)")
    parser.add_argument("--fps", type=float, default=5, help="Images per second of video (default: 5)")
    parser.add_argument("--blur", type=float, default=0,
                        help="Optional: skip frames with a sharpness score below this, e.g. 20. "
                             "Default 0 keeps every frame")
    args = parser.parse_args()

    videos = find_videos(args.input)
    if not videos:
        sys.exit(f"No videos found at: {args.input}")

    print(f"Found {len(videos)} video(s). Saving {args.fps} images/sec to '{args.out}'")
    print("Blur filter: " + (f"on (limit {args.blur})" if args.blur > 0 else "off (keeping all frames)") + "\n")
    total_saved = total_skipped = 0
    for v in videos:
        print(f"-> {os.path.basename(v)}")
        saved, skipped = extract(v, args.out, args.fps, args.blur)
        print(f"   saved {saved} images, skipped {skipped} blurry")
        total_saved += saved
        total_skipped += skipped

    print(f"\nDone. {total_saved} images saved, {total_skipped} blurry frames skipped.")
    print(f"Images are in: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
