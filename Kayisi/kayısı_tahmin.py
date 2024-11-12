import torch
from roboflow import Roboflow
import os
from ultralytics import YOLO  # YOLOv8 modülü

# CUDA ve GPU kullanılabilirliğini kontrol et
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

# Roboflow API anahtarı ile projeyi indir
rf = Roboflow(api_key="27NGLBsDmR81psmJIq4O")
project = rf.workspace("eslem-nbwqb").project("kayisi-7cnys")
version = project.version(1)
dataset = version.download("yolov8")  # YOLOv8'le uyumlu olarak veri setini indiriyoruz

# Eğitim işlemini başlat
if __name__ == "__main__":
    # Önceden eğitilmiş YOLOv8 modelini başlat
    model = YOLO("yolov8s.pt")
    
    # Sonuçların kaydedileceği klasörü ayarla
    results_folder = "results"  # Sonuç klasörü
    os.makedirs(results_folder, exist_ok=True)

    # Modeli eğit ve sonuçları belirtilen klasöre kaydet
    model.train(data=f"{dataset.location}/data.yaml", epochs=5, imgsz=640, batch=16, project=results_folder, name="train_results")
    print("Eğitim tamamlandı.")
