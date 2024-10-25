import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle (görüntü yolunu doğru şekilde ver)
image = cv2.imread('/Users/eslemnurgok/Desktop/kare.png', cv2.IMREAD_GRAYSCALE)

# Eğer görüntü yüklenemediyse hata mesajı ver
if image is None:
    print("Görüntü yüklenemedi, lütfen görüntü yolunu kontrol edin.")
    exit()

# X ve Y yönündeki gradyanları Sobel operatörü ile hesapla
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # ∂f / ∂x
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # ∂f / ∂y

# Gradyan büyüklüğünü hesapla (kenar gücü)
magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Gradyan yönünü hesapla (kenarın normal açısı)
angle = np.arctan2(grad_y, grad_x) * (180.0 / np.pi)

# Gradyan büyüklüğünü ve yönünü görselleştir ve kaydet
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Gradyan Büyüklüğü")
plt.imshow(magnitude, cmap='gray')
plt.axis('off')  # Eksenleri gizle
plt.savefig('gradyan_buyuklugu.png')  # Gradyan büyüklüğünü kaydet

plt.subplot(1, 2, 2)
plt.title("Gradyan Yönü")
plt.imshow(angle, cmap='gray')
plt.axis('off')  # Eksenleri gizle
plt.savefig('gradyan_yonu.png')  # Gradyan yönünü kaydet

plt.tight_layout()  # Daha iyi yerleşim
plt.show()

# Gradyan büyüklüğü ve yönünü ayrı ayrı kaydet
cv2.imwrite('/Users/eslemnurgok/Desktop/gradyan_buyuklugu_cv2.png', (magnitude * 255 / np.max(magnitude)).astype(np.uint8))
cv2.imwrite('/Users/eslemnurgok/Desktop/gradyan_yonu_cv2.png', (angle * 255 / np.max(angle)).astype(np.uint8))

print("Gradyan büyüklüğü ve yönü başarıyla kaydedildi.")
