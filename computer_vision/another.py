import cv2
import numpy as np

def detect_yellow_lines(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # bu sarının aralıkları
    lower_yellow = np.array([15, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    # sarı renk paleti 
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # kernel zart zurt
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    # Hough Transform
    yellow_lines = cv2.HoughLinesP(yellow_mask, 2, np.pi/180, 100, minLineLength=40, maxLineGap=5)

    # Sarı renk maskesi oluştur 
    yellow_lines_mask = np.zeros_like(image)
    if yellow_lines is not None:
        for line in yellow_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(yellow_lines_mask, (x1, y1), (x2, y2), (0, 255, 255), 5)

    # Sarı çizgileri orijinal görüntü üzerine çizdik
    result = cv2.addWeighted(image, 0.8, yellow_lines_mask, 1, 0)

    return result

# Video 
cap = cv2.VideoCapture('videonuz.mp4')  # Test Video dosyanızın adını buraya ekleyin

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Sarı çizgi algılama fonksiyonunu çağır
    result = detect_yellow_lines(frame)

    cv2.imshow('Yellow Lines Detection', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
