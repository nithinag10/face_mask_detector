# Face Mask Detector and alarmer
It is a machine learning application which watch out people's mask and alarm if they don't have one. 

## How to run the app

1. Either fork or download the app and open the folder in the cli.
2. Install all dependencies using `pip install requirements.txt` command.
3. To start the local sever on your machine Run `export FLASK_APP=app.py` `export FLASK_ENV=development` and the `flask run`. The app will start running on https://localhost:5000/ 
4. To use the service of the local host run `python test.py`


## How to train the machine learning model (optional)

There is already a trained model in the repo if you have to train it again you can do it.
1. Run `cd model_training`
2. Create two empty folder containing named 'train' and 'val' and create 'No mask' , 'proper' and 'not proper' directories in each of them.
3. Run `python load_images.py` this opens the webcam. You need to press the following keyboard keys to populate the training and validation images.
4. Press 's' and 't' by properly wearing mask in front of webcam to populate the images from cv2.
5. Press 'n' and 'y' by not wearing mask properly and capture the keys.
6. Press 'z' and 'u' by not wearing mask.
7. After making dataset run `python train_model.py`
8. After training run `cd ../` and `python test.py`

## Model uses Resnet18 architecture.

Transfer learning on resnet18 is performed using pytorch. Model with pretrainied weights are imported from pythorch and then modifying the last fc layer with 3 this model is trained again. This only affects the weights of the last Fully connected layer with changing the weights of the deep CNN.

## Output of the app

<img width="1440" alt="Screenshot 2021-05-02 at 11 27 46 AM" src="https://user-images.githubusercontent.com/56193327/116809266-c2e31380-ab5a-11eb-97ed-353a8272a0ff.png">
<img width="1440" alt="Screenshot 2021-05-02 at 11 28 18 AM" src="https://user-images.githubusercontent.com/56193327/116809338-21a88d00-ab5b-11eb-8f15-11a1ac554891.png">
 <img width="1440" alt="Screenshot 2021-05-02 at 11 26 52 AM" src="https://user-images.githubusercontent.com/56193327/116809321-089fdc00-ab5b-11eb-9c04-7fcf525c1b44.png">

## Conclusion

There are some complications when to comes to deploying in the real world. In this app, face recognition works fine which is built on resnet. But when it comes to face detection, haar_cascade from cv2 won't do the job greatly. If we use any advanced face detection technique we can definitely put it into real world usage. This can be implemented in CCTV in public places where is application has job.
It has nice alarm which rings if we don't wear a mask!. Try it.
