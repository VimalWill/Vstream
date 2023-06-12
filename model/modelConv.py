from ultralytics import YOLO

model = YOLO("/home/vimal/Edge AI/Vstream/model/yolov8n.pt")
#metric = model.val()
results = model("https://ultralytics.com/images/bus.jpg")
path = model.export(format="onnx")
print(path)