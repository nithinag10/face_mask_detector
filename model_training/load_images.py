import cv2

webcam = cv2.VideoCapture(0)
size = 4
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 14

while(True):
    (rval, im) = webcam.read()

    im = cv2.flip(im, 1, 1)
    # Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces
    faces = classifier.detectMultiScale(mini)
    for f in faces:
        (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
        # Save just the rectangle faces in SubRecFaces
        face_img = im[y:y + h, x:x + w]
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0,0), 2)

    cv2.imshow('LIVE', im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop
    if key == 27:  # The Esc key
        break
    elif key == ord('s'):
        cv2.imwrite(f'data/train/proper/{count}.jpg', face_img),
        count += 1
    elif key == ord('n'):
        cv2.imwrite(f'data/train/not proper/{count}.jpg', face_img)
        count += 1
    elif key == ord('z'):
        cv2.imwrite(f'data/train/No mask/{count}.jpg', face_img)
        count += 1
    elif key == ord('t'):
        cv2.imwrite(f'data/val/proper/{count}.jpg', face_img),
        count += 1
    elif key == ord('y'):
        cv2.imwrite(f'data/val/not proper/{count}.jpg', face_img)
        count += 1
    elif key == ord('u'):
        cv2.imwrite(f'data/val/No mask/{count}.jpg', face_img)
        count += 1


webcam.release()
cv2.destroyAllWindows()




