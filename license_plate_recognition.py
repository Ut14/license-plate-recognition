import sys
import torch
import cv2
import numpy as np
import easyocr
import gradio as gr
from PIL import Image, ImageDraw

# Ensure YOLOv9 is accessible
sys.path.append("D:/ml/yolov9")  # Update this path if needed

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords

# Load YOLO model
MODEL_PATH = "best.pt"  # Update if using a different model path
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DetectMultiBackend(MODEL_PATH, device=device)
model.eval()

# Load EasyOCR
reader = easyocr.Reader(['en'])  # English OCR

def detect_license_plate(image):
    """ Detect license plate and extract text """
    
    # Convert to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Preprocess for YOLOv9
    img = torch.from_numpy(img_cv).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)

    # Run YOLO detection
    with torch.no_grad():
        pred = model(img)
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.4)[0]

    # Process detections
    if pred is None or len(pred) == 0:
        return image, "No license plate detected."

    # Convert to original image dimensions
    pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img_cv.shape).round()

    # Draw bounding boxes and perform OCR
    draw = ImageDraw.Draw(image)
    extracted_texts = []

    for *xyxy, conf, cls in pred:
        x1, y1, x2, y2 = map(int, xyxy)

        # Crop detected plate
        plate_img = img_cv[y1:y2, x1:x2]

        # OCR recognition
        text = reader.readtext(plate_img, detail=0)
        extracted_texts.append(" ".join(text))

        # Draw box and text
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        draw.text((x1, y1 - 10), " ".join(text), fill="red")

    return image, "\n".join(extracted_texts)

# Gradio Interface
interface = gr.Interface(
    fn=detect_license_plate,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil"), gr.Text()],
    title="License Plate Recognition",
    description="Upload an image, and the system will detect and read the license plate."
)

if __name__ == "__main__":
    interface.launch()
