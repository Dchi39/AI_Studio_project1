import google.generativeai as genai
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from segment_anything import SamPredictor, sam_model_registry
import os

# === Step 1: Gemini to identify object ===
genai.configure(api_key="API KEY")  # Replace with your API key
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# Load and upload image to Gemini
image_path = "img/remote2.jpg"  # Replace with your image
uploaded_file = genai.upload_file(image_path)
response = model.generate_content([uploaded_file, "Detect the main object."])
print("Gemini Caption:", response.text)  # e.g., "A black pen on a white background"

# === Step 2: Load image locally ===
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# === Step 3: Use SAM to segment object ===
# Load SAM model
sam_checkpoint = "weight/sam_vit_b_01ec64.pth"  # Download from Meta's repo
sam = sam_model_registry["vit_b"](checkpoint="weight/sam_vit_b_01ec64.pth")
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

# Prepare image
predictor.set_image(image_rgb)

# Use center point to prompt (you can use Gemini for better object location!)
h, w, _ = image.shape
input_point = np.array([[w//2, h//2]])
input_label = np.array([1])

masks, scores, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
mask = masks[0].astype(np.uint8)

# === Step 4: PCA for orientation ===
def get_orientation(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    data_pts = np.squeeze(main_contour).astype(np.float64)
    mean = np.mean(data_pts, axis=0)
    centered = data_pts - mean
    cov = np.cov(centered.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    order = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, order]
    angle = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))
    return angle, mean, eig_vecs

angle, center, eig_vecs = get_orientation(mask * 255)

# === Step 5: Draw results ===
output = image.copy()
center = tuple(map(int, center))
axis_end = (
    int(center[0] + 100 * eig_vecs[0, 0]),
    int(center[1] + 100 * eig_vecs[1, 0])
)
cv2.line(output, center, axis_end, (0, 0, 255), 3)
cv2.putText(output, f"Angle: {angle:.2f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

# Overlay mask
colored_mask = cv2.merge([mask*150]*3)
seg_overlay = cv2.addWeighted(output, 0.8, colored_mask, 0.6, 0)

# === Step 6: Plot ===
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB))
plt.title(f"Segmentation + Orientation\nAngle: {angle:.2f}°")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.plot([0, np.cos(np.radians(angle))], [0, np.sin(np.radians(angle))], 'r-', linewidth=3)
plt.title("Orientation Vector")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([-1, 1])
plt.ylim([-1, 1])

plt.tight_layout()
plt.show()
