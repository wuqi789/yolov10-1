from ultralytics import YOLO

# load a pretrained model (recommended for training)
model = YOLO('E:/yolov10-main/runs/train/exp2/weights/last.pt')

# Export the model
success = model.export(format='onnx')

print(success)
