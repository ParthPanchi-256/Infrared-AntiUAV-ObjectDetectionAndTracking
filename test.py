from ultralytics import YOLO
from pathlib import Path
import cv2
import torch
import re


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11n.pt"

# Folder containing:
# 001481.jpg
# 001482.jpg
# 001483.jpg
# ...
INPUT_DIR = r"C:\Users\parth\Downloads\track1_test\track1_test\wg2022_ir_054_split_03"

# Output video
OUTPUT_VIDEO = "anti_uav_tracked.mp4"

# Output FPS
# Change this to match the original video FPS
FPS = 30

# ByteTrack configuration
TRACKER = "bytetrack.yaml"

# Detection confidence
CONFIDENCE = 0.25

# IoU threshold
IOU = 0.5

# Save individual annotated frames?
SAVE_FRAMES = False

# Folder for annotated frames
OUTPUT_FRAMES_DIR = "tracked_frames"


# ============================================================
# GPU CHECK
# ============================================================

if torch.cuda.is_available():

    DEVICE = 0

    print("=" * 60)
    print("CUDA AVAILABLE")
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print("=" * 60)

else:

    DEVICE = "cpu"

    print("=" * 60)
    print("WARNING: CUDA NOT AVAILABLE")
    print("Running on CPU")
    print("=" * 60)


# ============================================================
# PATHS
# ============================================================

input_dir = Path(INPUT_DIR)

if not input_dir.exists():
    raise FileNotFoundError(
        f"Input folder does not exist: {input_dir.resolve()}"
    )


if SAVE_FRAMES:

    output_frames_dir = Path(OUTPUT_FRAMES_DIR)
    output_frames_dir.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# GET IMAGE FILES
# ============================================================

image_files = list(input_dir.glob("*.jpg"))

# Also support JPG/JPEG/PNG
image_files += list(input_dir.glob("*.JPG"))
image_files += list(input_dir.glob("*.jpeg"))
image_files += list(input_dir.glob("*.JPEG"))
image_files += list(input_dir.glob("*.png"))


if len(image_files) == 0:

    raise RuntimeError(
        f"No image files found in {input_dir.resolve()}"
    )


# ============================================================
# NATURAL / NUMERICAL SORT
# ============================================================

def numerical_sort(path):

    """
    Extract number from filename.

    Example:

    001481.jpg -> 1481
    001482.jpg -> 1482
    """

    numbers = re.findall(
        r"\d+",
        path.stem
    )

    if numbers:
        return int(numbers[-1])

    return 0


image_files.sort(
    key=numerical_sort
)


print()
print("Input folder:")
print(input_dir.resolve())

print()
print("Number of frames:", len(image_files))

print()
print("First frame:", image_files[0].name)
print("Last frame :", image_files[-1].name)

print()


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded.")
print()


# ============================================================
# READ FIRST FRAME
# ============================================================

first_frame = cv2.imread(
    str(image_files[0])
)

if first_frame is None:

    raise RuntimeError(
        f"Could not read first frame: {image_files[0]}"
    )


height, width = first_frame.shape[:2]

print("Video resolution:")
print(f"{width} x {height}")

print("FPS:", FPS)

print()


# ============================================================
# CREATE VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    FPS,
    (width, height)
)


if not video_writer.isOpened():

    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# PROCESS FRAMES
# ============================================================

print("=" * 60)
print("STARTING DETECTION + TRACKING")
print("=" * 60)


for frame_index, image_path in enumerate(image_files):

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    frame = cv2.imread(
        str(image_path)
    )

    if frame is None:

        print(
            f"WARNING: Could not read {image_path.name}"
        )

        continue


    # --------------------------------------------------------
    # YOLO DETECTION + TRACKING
    # --------------------------------------------------------

    results = model.track(

        frame,

        tracker=TRACKER,

        persist=True,

        conf=CONFIDENCE,

        iou=IOU,

        device=DEVICE,

        verbose=False
    )


    result = results[0]


    # --------------------------------------------------------
    # Draw detections and tracking IDs
    # --------------------------------------------------------

    annotated_frame = result.plot()


    # --------------------------------------------------------
    # Write frame to video
    # --------------------------------------------------------

    video_writer.write(
        annotated_frame
    )


    # --------------------------------------------------------
    # Optionally save annotated frame
    # --------------------------------------------------------

    if SAVE_FRAMES:

        output_frame_path = (
            Path(OUTPUT_FRAMES_DIR)
            / image_path.name
        )

        cv2.imwrite(
            str(output_frame_path),
            annotated_frame
        )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        frame_index % 50 == 0
        or frame_index == len(image_files) - 1
    ):

        percentage = (
            (frame_index + 1)
            / len(image_files)
            * 100
        )

        print(
            f"[{frame_index + 1:5d}/"
            f"{len(image_files)}] "
            f"{percentage:6.2f}% "
            f"| {image_path.name}"
        )


# ============================================================
# CLEANUP
# ============================================================

video_writer.release()

cv2.destroyAllWindows()


# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("TRACKING COMPLETE")
print("=" * 60)

print(
    "Output video:"
)

print(
    Path(OUTPUT_VIDEO).resolve()
)

print()