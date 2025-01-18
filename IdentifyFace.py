import cv2

# Tải Haar Cascade
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Kiểm tra xem tệp có được tải thành công không
if haar_cascade.empty():
    print("Error loading cascade file")
    exit()

# Khởi tạo camera và xử lý hình ảnh
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(grayImg, scaleFactor=1.3, minNeighbors=4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(10) == 27:  # Nhấn Esc để thoát
        break

cam.release()
cv2.destroyAllWindows()
