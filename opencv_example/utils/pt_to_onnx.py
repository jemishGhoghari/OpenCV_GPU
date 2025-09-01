from ultralytics import YOLO

# Load the YOLO12 model
model = YOLO("../models/yolo12s.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo12s.onnx'

# Load the exported ONNX model
onnx_model = YOLO("../models/yolo12s.onnx")