import torch
import cv2
import numpy as np

# YOLOv5 modelini indiriyoruz
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Deneme amaçlı bir resim dosyasını oku
img = cv2.imread('/Users/eslemnurgok/Desktop/images.jpeg')  # Test edeceğin görüntüyü ekle

# Resmi YOLO modeline gönder
results = model(img)

# Sonuçları elde et
detections = results.xyxy[0]  # Tespit edilen nesneler

# Çerçeveleri çiz ve etiketleri ekle
for *xyxy, conf, cls in detections:
    label = f"{model.names[int(cls)]} {conf:.2f}"  # Etiket ve güven puanı
    cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)  # Çerçeve çiz
    cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Etiketi yaz

# Sonuç görüntüsünü kaydet
output_path = '/Users/eslemnurgok/Desktop/images_output2.jpeg'  # Farklı bir dosya adı belirt
cv2.imwrite(output_path, img)

# Sonuçları göster
cv2.imshow('Nesne Tespiti', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
