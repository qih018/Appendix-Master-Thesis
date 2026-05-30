import os
from pathlib import Path
from datetime import datetime, timedelta

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import imageio.v2 as imageio
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont


# -----------------------------------------------------
# USER INPUT
# -----------------------------------------------------
OUTPUT_ROOT  = Path(r"C:\Users\rakel\OneDrive - University of Bergen\Master\NaCl exp")
#DICOM_FOLDER = Path(r"C:\Users\rakel\OneDrive - University of Bergen\Master\NaCl_CO2_hydrates_pt2\37\pdata\1\dicom")
DICOM_FOLDER = Path(r"C:\Users\rakel\OneDrive - University of Bergen\Master\CO2_hydrates_part1\2025_06_CO2_hydrate\MRI_scan_data\55\pdata\1\dicom")
FIRST_SLICE_NUMBER = 15
#STEP = 25
STEP = 33

N_FRAMES= 50
#N_FRAMES = 60

UPSCALE_FACTOR = 8
fps = 5

TR_MS = 4000
TR_SECONDS = TR_MS / 1000
SECONDS_PER_EXPORTED_FRAME = STEP * TR_SECONDS


# Injection start time
#REFERENCE_TIME = datetime.strptime("24.06.2025 16:57:00", "%d.%m.%Y %H:%M:%S")
REFERENCE_TIME = datetime.strptime("02.12.2025 13:32:00", "%d.%m.%Y %H:%M:%S")


# -----------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------
def pad_to_mod16(img):
    h, w = img.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')


def add_text_under_image(img_8bit, text, strip_height=50):
    pil_img = Image.fromarray(img_8bit)
    w, h = pil_img.size

    canvas = Image.new("L", (w, h + strip_height), color=255)
    canvas.paste(pil_img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except:
        text_w, text_h = draw.textsize(text, font=font)

    x = (w - text_w) // 2
    y = h + (strip_height - text_h) // 2

    draw.text((x, y), text, fill=0, font=font)
    return np.array(canvas)


# -----------------------------------------------------
# LOAD DICOM FILES
# -----------------------------------------------------
dicom_files = sorted(DICOM_FOLDER.glob("*.dcm"),
                     key=lambda p: p.name.lower())

print(f"Found {len(dicom_files)} DICOMs")

if FIRST_SLICE_NUMBER >= len(dicom_files):
    raise ValueError("FIRST_SLICE_NUMBER outside DICOM stack")

indices = [
    FIRST_SLICE_NUMBER + i * STEP
    for i in range(N_FRAMES)
    if FIRST_SLICE_NUMBER + i * STEP < len(dicom_files)
]

print(f"Using {len(indices)} frames")
print(f"Frame spacing = {SECONDS_PER_EXPORTED_FRAME:.2f} seconds")


# -----------------------------------------------------
# ROI SELECTION
# -----------------------------------------------------
ref_ds = pydicom.dcmread(dicom_files[FIRST_SLICE_NUMBER])
ref_img = ref_ds.pixel_array.astype(float)

roi_coords = {}

def line_select_callback(eclick, erelease):
    roi_coords['x1'], roi_coords['y1'] = int(eclick.xdata), int(eclick.ydata)
    roi_coords['x2'], roi_coords['y2'] = int(erelease.xdata), int(erelease.ydata)

fig, ax = plt.subplots()
ax.imshow(ref_img, cmap="gray")
ax.set_title("Draw ROI then close window")
toggle_selector = RectangleSelector(
    ax,
    line_select_callback,
    interactive=True,
    useblit=True,
    button=[1]
)

plt.show()

if not roi_coords:
    raise RuntimeError("No ROI drawn!")

x1, x2 = sorted([roi_coords['x1'], roi_coords['x2']])
y1, y2 = sorted([roi_coords['y1'], roi_coords['y2']])

mask = np.zeros_like(ref_img, dtype=np.uint8)
mask[y1:y2, x1:x2] = 1


# -----------------------------------------------------
# OUTPUT FOLDER
# -----------------------------------------------------
out_dir = OUTPUT_ROOT / f"Pngs_for_slice_{FIRST_SLICE_NUMBER}"
out_dir.mkdir(exist_ok=True)


# -----------------------------------------------------
# PROCESS IMAGES
# -----------------------------------------------------
for idx, dcm_index in enumerate(indices):

    ds = pydicom.dcmread(dicom_files[dcm_index])
    img = ds.pixel_array.astype(float)

    roi = img * mask

    roi_vals = roi[mask == 1]

    if len(roi_vals) > 0 and roi_vals.max() > roi_vals.min():

        roi_scaled = np.zeros_like(roi)

        roi_scaled[mask == 1] = (
            (roi_vals - roi_vals.min())
            / (roi_vals.max() - roi_vals.min())
            * 65535
        )

        roi_scaled = roi_scaled.astype(np.uint16)

    else:
        roi_scaled = roi.astype(np.uint16)


    if UPSCALE_FACTOR > 1:

        img_resized = resize(
            roi_scaled,
            (
                roi_scaled.shape[0] * UPSCALE_FACTOR,
                roi_scaled.shape[1] * UPSCALE_FACTOR
            ),
            preserve_range=True,
            anti_aliasing=False,
            order=1
        ).astype(np.uint16)

    else:
        img_resized = roi_scaled


    img_resized = pad_to_mod16(img_resized)

    img_8bit = (img_resized / 256).astype(np.uint8)


    # -------------------------------------------------
    # TIME LABEL (minutes after injection)
    # -------------------------------------------------
   # -------------------------------------------------
# TIME LABEL BASED ON DICOM ACQUISITION TIME
# -------------------------------------------------
acq_time_str = ds.AcquisitionTime  # e.g. "133214.532000"

# Remove fractional seconds if present
acq_time_str_main = acq_time_str.split('.')[0]

# Parse time part from DICOM
acq_time_only = datetime.strptime(acq_time_str_main, "%H%M%S")

# Combine with the same date as REFERENCE_TIME
acq_time = REFERENCE_TIME.replace(
    hour=acq_time_only.hour,
    minute=acq_time_only.minute,
    second=acq_time_only.second,
    microsecond=0
)

# If scan passed midnight relative to reference, adjust if needed
if acq_time < REFERENCE_TIME - timedelta(hours=12):
    acq_time += timedelta(days=1)
elif acq_time > REFERENCE_TIME + timedelta(hours=12):
    acq_time -= timedelta(days=1)

elapsed_seconds = (acq_time - REFERENCE_TIME).total_seconds()
elapsed_minutes = abs(elapsed_seconds) / 60

if elapsed_seconds < 0:
    label = f"{elapsed_minutes:.2f} min before injection"
else:
    label = f"{elapsed_minutes:.2f} min after injection"


    img_with_text = add_text_under_image(img_8bit, label)

    out_path = out_dir / f"Rep_{idx:03d}.png"

    imageio.imwrite(out_path, img_with_text)

    print(f"Saved: {out_path} | {label}")


print("All frames saved.")


# -----------------------------------------------------
# CREATE VIDEO
# -----------------------------------------------------
video_path = OUTPUT_ROOT / f"slice_{FIRST_SLICE_NUMBER}_dynamic.mp4"

files = sorted(out_dir.glob("*.png"))

with imageio.get_writer(
        video_path,
        fps=fps,
        codec='libx264',
        format='ffmpeg',
        ffmpeg_params=[
            '-pix_fmt', 'yuv420p',
            '-crf', '12',
            '-preset', 'slow',
            '-b:v', '50M'
        ]) as writer:

    for f in files:
        writer.append_data(imageio.imread(f))

print(f"Video saved to: {video_path}")