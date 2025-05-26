from google import genai
from google.genai import types
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json

# Initialize the client
client = genai.Client(api_key="ADD API KEY HERE")

# Load and upload image1
image1_path = "img/real.jpg"
uploaded_file = client.files.upload(file=image1_path)

# Load image2 as bytes
image2_path = "img/test.jpg"
with open(image2_path, 'rb') as f:
    img2_bytes = f.read()

# Send prompt to Gemini model
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        "Find the defect of image two comparing image one and Detect the all of the prominent items in the image. "
        "The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000 and return them as a JSON list.",
        uploaded_file,
        types.Part.from_bytes(
            data=img2_bytes,
            mime_type='image/png'
        )
    ]
)

# Print and parse the response
print("Model Output:\n", response.text)

# Try to extract JSON bounding box from model response
try:
    # Find JSON in the response text
    bbox_list = json.loads(response.text.strip().split('```json')[-1].split('```')[0])
except Exception as e:
    print("Error parsing JSON:", e)
    bbox_list = []

print("Parsed bounding boxes:", bbox_list)

# Load test image for drawing
image = Image.open(image2_path)
draw = ImageDraw.Draw(image)

# Get image size
width, height = image.size

# Draw each bounding box
for item in bbox_list:
    if 'box_2d' in item and len(item['box_2d']) == 4:
        ymin, xmin, ymax, xmax = item['box_2d']
        # Normalize 0–1000 to pixel size
        left = int(xmin / 1000 * width)
        top = int(ymin / 1000 * height)
        right = int(xmax / 1000 * width)
        bottom = int(ymax / 1000 * height)

        # Optionally show label
        label = item.get('label', 'object')
        draw.rectangle([left, top, right, bottom], outline="yellow", width=10)
        draw.text((left, top), label, fill="red")
    else:
        print("Invalid bounding box entry:", item)

# Show image with bounding boxes
plt.imshow(image)
plt.axis("off")
plt.title("Detected Bounding Boxes")
plt.show()
