#Face Mask Detector and alarmer
It is a machine learning application which watch out people's mask and alarm if they don't have one. 

## How to run the app

1.Either fork or download the app and open the folder in the cli
2.Install all dependencies using `pip install requirements.txt` command
3.To start the local sever on your machine set up flask environment and app
4.Run `export FLASK_APP=app.py` `export FLASK_ENV=development` and the `flask run`. The app will start running on https://localhost:5000/
5.To use the service of the local host run `python test.py`

## How to train the machine learning model (optional)

There is already a trained model in the repo if you have to train it again you can do it.
1. Run 'cd model_training'
2. Create two empty folder containing named 'train' and 'val' and create 'No mask' , 'proper' and 'not proper' directories in each of them.
3. Run `python load_images.py` this opens the webcam. You need to press the following keyboard keys to populate the training and validation images.
4. Press 's' and 't' by properly wearing mask in front of webcam to populate the images from cv2.
5. Press 'n' and 'y' by not wearing mask properly and capture the keys.
6. Press 'z' and 'u' by not wearing mask.
7. After making dataset run `python train_model.py`
8. After training run `cd ../` and `python test.py`

## Model trained using ResNet18

The model is training on pretrained resnet model. The model is modified in the last fc layer with no of output we are wishing to have. In our case its 3. When the model is trained only the weights of the last fc layer changes.
