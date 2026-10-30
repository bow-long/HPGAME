
from ultralytics import YOLO

# Load your existing trained model
model = YOLO(r"runs/detect/blue_led/weights/best.pt")

# Export for CPU inference.
# dynamic=True is useful because your game uses different input sizes.
exported_path = model.export(
    format="openvino",
    dynamic=True,
    imgsz=416,
)

print("Export complete!")
print("Exported model:", exported_path)
