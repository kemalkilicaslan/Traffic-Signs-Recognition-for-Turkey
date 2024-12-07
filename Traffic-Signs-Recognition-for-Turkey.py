from ultralytics import YOLO

model = YOLO('Traffic_Signs_Recognition_for_Turkey.pt')
results = model('Traffic_Signs_Recognition_for_Turkey.mp4', save=True)