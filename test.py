import requests
import cv2
import simpleaudio as sa
import time

size = 4
webcam = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
color_dict={0:(255,0,0),1:(0,255,0) , 2:(0,0,255)} #color codes for frames
labels_dict={0:'with_out mask',1:'with_mask' , 2:'Wear Properly'}
count = 0


while(True):
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,1)
    # Flip to act as a mirror
    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
    # detect MultiScale / faces
    faces = classifier.detectMultiScale(mini)
    for f in faces:
        (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
        # Save just the rectangle faces in SubRecFaces
        face_img = im[y:y + h, x:x + w]
        content_type = 'image/jpeg'
        headers = {'content-type': content_type}
        _, img_encoded = cv2.imencode('.jpg',face_img)
        # send http request with image
        # array is converted into bytes to transmit to the server
        response = requests.post("http://localhost:5000/predict", data=img_encoded.tobytes(),headers=headers)
        label = response.json()['prediction']
        if label == 0:
            count += 1
            if count >= 10:
                # timed loop for 4 sec
                start_time = time.time()
                seconds = 4
                while True:
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    filename = 'buzzer.wav'
                    wave_obj = sa.WaveObject.from_wave_file(filename)
                    play_obj = wave_obj.play()
                    play_obj.wait_done()
                    # Wait until sound has finished playing
                    count = 0
                    if elapsed_time > seconds:
                        break
        cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop
    if key == 27:  # The Esc key
        break

webcam.release()
# Close all started windows
cv2.destroyAllWindows()

