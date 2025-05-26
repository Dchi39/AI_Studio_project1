from google import genai
from google.genai import types
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

# Initialize Gemini client
client = genai.Client(api_key="ADD API KEY HERE")

# Upload reference image
ref_img_path = "img/real.jpg"
uploaded_file = client.files.upload(file=ref_img_path)

# Load test image to analyze
image2_path = "img/test.jpg"
with open(image2_path, 'rb') as f:
    img2_bytes = f.read()

# Ask Gemini for bounding boxes
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        "Compare image 2 with image 1. Detect prominent items in image 2. "
        "Return a JSON list with 'label' and 'box_2d' = [ymin, xmin, ymax, xmax] normalized from 0–1000.",
        uploaded_file,
        types.Part.from_bytes(data=img2_bytes, mime_type='image/jpeg')
    ]
)

# Parse JSON response
try:
    bbox_list = json.loads(response.text.strip().split('```json')[-1].split('```')[0])
except Exception as e:
    print("Failed to parse Gemini output:", e)
    bbox_list = []

# Load image with OpenCV
image_cv = cv2.imread(image2_path)
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
height, width = image_rgb.shape[:2]

# Draw horizontal midline
mid_y = height // 2
cv2.line(image_rgb, (0, mid_y), (width, mid_y), (0, 255, 255), 2)

for item in bbox_list:
    if 'box_2d' in item:
        ymin, xmin, ymax, xmax = item['box_2d']
        # Convert from 0–1000 to pixel coordinates
        left = int(xmin / 1000 * width)
        top = int(ymin / 1000 * height)
        right = int(xmax / 1000 * width)
        bottom = int(ymax / 1000 * height)

        # Crop ROI
        roi = image_rgb[top:bottom, left:right]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Largest contour
            contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(contour)  # (center, (w,h), angle)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            box[:, 0] += left
            box[:, 1] += top

            # Draw rotated rectangle
            cv2.drawContours(image_rgb, [box], 0, (0, 255, 0), 3)

            # Extract long side center line
            (cx, cy), (w, h), angle = rect
            cx += left
            cy += top

            if h > w:
                long_len = h
                angle_adjusted = angle + 90
            else:
                long_len = w
                angle_adjusted = angle

            # Draw long axis line
            dx = int(np.cos(np.deg2rad(angle_adjusted)) * long_len / 2)
            dy = int(np.sin(np.deg2rad(angle_adjusted)) * long_len / 2)
            pt1 = (int(cx - dx), int(cy - dy))
            pt2 = (int(cx + dx), int(cy + dy))
            cv2.line(image_rgb, pt1, pt2, (255, 0, 0), 2)

            # Calculate angle to horizontal midline (only horizontal part matters)
            delta_x = pt2[0] - pt1[0]
            delta_y = pt2[1] - pt1[1]
            angle_to_horizontal = np.degrees(np.arctan2(delta_y, delta_x))

            # Display angle
            angle_text = f"Angle: {angle_to_horizontal:.1f}°"
            cv2.putText(image_rgb, angle_text, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(image_rgb, item.get("label", "Object"), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Show result
plt.imshow(image_rgb)
plt.title("Object Orientation Relative to Horizontal Midline")
plt.axis("off")
plt.show()
