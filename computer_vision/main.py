import cv2
import numpy as np

def detect_lane(image):
    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Kenar tespiti yapıyoruz canny kullanarak
    edges = cv2.Canny(gray, 50, 150)

    # ROI (Region of Interest) belirlenen kısım 
    mask = np.zeros_like(edges)
    height, width = edges.shape
    polygon = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough Transform'u uygula yol üzerinde noktaları çıkardık
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, minLineLength=40, maxLineGap=5)

    # Elde edilen çizgileri orijinal görüntü üzerine çiz
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Orijinal görüntü ile çizgileri birleştir
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return result

# Video akışı aldığım kısım
cap = cv2.VideoCapture('videonuz.mp4')  # Video veya fotoğrafı ekle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Şerit algıladığımız fonksiyonu buradan çağırıyorum
    result = detect_lane(frame)

    # Sonucu göster
    cv2.imshow('Lane Detection', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
