from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11m.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11m.onnx'

# Load the exported ONNX model
onnx_model = YOLO("yolo11m.onnx")