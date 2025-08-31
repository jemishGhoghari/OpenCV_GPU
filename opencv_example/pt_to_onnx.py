from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo12m.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11x.onnx'

# Load the exported ONNX model
onnx_model = YOLO("yolo12m.onnx")