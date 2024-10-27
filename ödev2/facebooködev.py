import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
import requests
import torch
from torchvision import transforms
import cv2

def initialize_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    return driver

# Facebook'a giriş yapma alanı
def login_facebook(driver, email, password):
    driver.get("https://www.facebook.com/")
    time.sleep(2)
    
    driver.find_element(By.ID, "email").send_keys(email)
    driver.find_element(By.ID, "pass").send_keys(password)
    driver.find_element(By.NAME, "login").click()
    time.sleep(5)

# Facebook profil fotoğraflarını toplayan kısım
def get_profile_pictures(driver, profile_url, save_folder="downloaded_images", num_photos=10):
    os.makedirs(save_folder, exist_ok=True)
    driver.get(profile_url)
    time.sleep(5)
    
    images = driver.find_elements(By.TAG_NAME, 'img')
    img_links = [img.get_attribute('src') for img in images if img.get_attribute('src')]
    img_links = img_links[:num_photos]
    
    print(f"Toplam indirilecek görsel sayısı: {len(img_links)}")
    for i, img_url in enumerate(img_links):
        img_data = requests.get(img_url).content
        with open(f"{save_folder}/profile_{i}.jpg", 'wb') as handler:
            handler.write(img_data)
        print(f"Fotoğraf indirildi: profile_{i}.jpg")

# YOLO ile insan tespiti yapan kısım
def detect_humans(image_path, model):
    image = cv2.imread(image_path)
    results = model(image)
    detected = results.xyxy[0]  
    for *box, conf, cls in detected:
        if int(cls) == 0 and conf > 0.5:  # 0, COCO veri kümesinde 'insan' sınıfı
            return True  # İnsan tespit edildi
    return False  # İnsan tespit edilmedi

def main():
    # Kullanıcı adı ve şifre bilgilerini girdiğimiz alan
    email = "Eslem Gök"
    password = "Yalova898846"
    profile_url = "https://www.facebook.com/friends"  # Resimleri alıcağı arkadaşlar sayfası
    save_folder = "downloaded_images"
    
    # Sonuçları tutmak için bir liste oluşturuyoruz
    results = []

    # Tarayıcı başlatma ve giriş yapma alanı
    driver = initialize_driver()
    login_facebook(driver, email, password)
    get_profile_pictures(driver, profile_url, save_folder)
    driver.quit()
    
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 modelini yüklediğimiz alan

    # Fotoğraflarda insan tespiti yaptığımız yer
    for image_file in os.listdir(save_folder):
        if image_file.endswith('.jpg') or image_file.endswith('.jpeg'):
            image_path = os.path.join(save_folder, image_file)
            if detect_humans(image_path, model):
                result = f"{image_file} insan içeriyor."
            else:
                result = f"{image_file} insan içermiyor."
            results.append(result)  # Sonucu listeye ekle

    # Sonuçları Txt dosyasına yazdırma işlemi yapılan alan
    with open('results.txt', 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')  # Her sonucu yeni bir satıra yaz

if __name__ == "__main__":
    main()