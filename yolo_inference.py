from ultralytics import YOLO 

model = YOLO(r'C:\Users\Motas\OneDrive\Desktop\final pres\football_analysis-main\models\best.pt')

results = model.predict(r'C:\Users\Motas\OneDrive\Desktop\final pres\football_analysis-main\input_videos\08fd33_4.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)