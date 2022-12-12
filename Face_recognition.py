import cv2

face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(r'haarcascade_smile.xml')

cap = cv2.VideoCapture("peoples.mp4")

while True:
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        rec_gray = gray_img[y:y + h, x:x + w]
        rec_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(rec_gray)
        smile = smile_cascade.detectMultiScale(rec_gray)

        for (sx, sy, sw, sh) in eyes:
            cv2.rectangle(rec_color, (sx, sy), (sx + sw, sy + sh), (0, 127, 255), 2)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(rec_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)

    cv2.imshow('Face Recognition', img)
    print(f'Found {len(faces)} people!')
    # cv2.putText(img, f'Found {len(faces)} people!', (x + w, y + h), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
