import os
from ultralytics import YOLO  # YOLOv8 modülü
from pathlib import Path

# Eğitilmiş modeli yükle (eğitim sırasında kaydedilen 'best.pt' dosyasını kullan)
model = YOLO("results/train_results/weights/best.pt")  # Eğitilmiş modelin yolunu belirtin

# Test klasörünün yolunu belirle
test_images_folder = "kayisi-1/test/images"  # Test görüntülerinin bulunduğu klasörün yolu
results_folder = "prediction_results"  # Tahmin sonuçlarının kaydedileceği klasör
os.makedirs(results_folder, exist_ok=True)

# Test klasöründeki her bir görüntüde tahmin yap ve kaydet
for idx, image_file in enumerate(os.listdir(test_images_folder), start=1):
    image_path = os.path.join(test_images_folder, image_file)

    # Tahminleri yap
    results = model.predict(source=image_path, conf=0.45, iou=0.45)

    # Tahmin edilen görüntüyü kaydet
    result_path = os.path.join(results_folder, f"prediction_{idx}.jpg")  # Burada os.path.join kullanıyoruz
    results[0].save(result_path)  # Kaydetme işlemi
    print(f"Processed {image_file} - Result saved as {result_path} in 'prediction_results' folder")
