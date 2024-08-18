# E:\yolov10-main\ultralytics\cfg\models\v10
# E:\yolov10-main\datasets
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLOv10

model_yaml_path = r'E:\yolov10-main\ultralytics\cfg\models\v10\yolov10n.yaml'

data_yaml_path = r'E:\yolov10-main\datasets\data.yaml'

if __name__ == '__main__':
    model = YOLOv10(model_yaml_path)
    results = model.train(data=data_yaml_path,
                          epochs=3,
                          batch=32,
                          workers=0,
                          device=0,
                          project='runs/train',
                          name='exp',
                          )
