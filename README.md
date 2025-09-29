# Detect Car Number Plate & Extract the Number

This project is about detecting number plate of a car in an image and then extracting the number from the plate. <br>
**Tensorflow** and **OpenCV** are used for training model and for image processing.

## Model

Here model used for Object Detection is **YOLOv4** from DarkNet. Used **AWS Textract** for extracting number from cropped image. <br>
The accuracy achieved by YOLOv4 was **98.9%**.

## Deployment

**Flask** is used to develop Web App along with HTML and CSS. Have also used **Docker** to containerize flask app for deployment. <br>
Dockerized flask app can be deployed on **Heroku**, **AWS**, **Google Cloud**, or **Azure**.

➤ In main page first we upload the image.
➤ On clicking **Detect** image will processed and the number will be extracted.

ouput 1:
![GUI SS1](https://github.com/user-attachments/assets/8386c019-8965-44e3-ae92-2e16874e239a)


output 2:
![GUI SS2](https://github.com/user-attachments/assets/02e868fb-e5ba-4bed-b487-29f750be6524)


