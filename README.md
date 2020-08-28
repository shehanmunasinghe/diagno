# Diagno - AI Based Remote Cardiology Solution

[![video](https://i.vimeocdn.com/filter/overlay?src0=https%3A%2F%2Fi.vimeocdn.com%2Fvideo%2F946614890_1280x720.webp&src1=https%3A%2F%2Ff.vimeocdn.com%2Fimages_v6%2Fshare%2Fplay_icon_overlay.png)](https://vimeo.com/451631301)

## [Visit Diagno Web App](http://diagno-ui.herokuapp.com/) 
## [Test Instructions](https://github.com/shehanmunasinghe/diagno/blob/master/Test%20Instructions.md)


## Inspiration

This project was started with the ultimate goal of applying modern technology to improve healthcare conditions across the globe and help save human lives.

![The Problem We Are Solving](https://github.com/shehanmunasinghe/diagno/blob/master/Docs/images/2.png?raw=true)


Cardiovascular diseases account for 17.9 million lives annually becoming the most common cause of human death. More than 75% of these deaths take place in low and middle-income countries, where people have limited access to healthcare resources and trained cardiologists. 

12-lead Electrocardiograms are among the major tools used by cardiologists for diagnosis of different heart conditions. The capturing of these signals usually happen through an ECG device. Ever since the first ECG device was invented, the process has been unchanged for decades, and accurate diagnosis heavily depends on well-trained cardiologists.

With the advent of deep neural networks, frameworks like PyTorch and large open-source datasets, there’s green-light that this process can be automated making healthcare solutions more affordable and accessible to everyone on earth

## What it does

![Web app screenshot 2](https://github.com/shehanmunasinghe/diagno/blob/master/Docs/images/Screenshot2.PNG?raw=true)

At Diagno we have developed a deep learning algorithm that is able to identify 5 different cardiac conditions from 12-lead ECG signals with over 90% accuracy. Diagno's web app [http://diagno-ui.herokuapp.com/] allows anyone to upload a 12-lead ECG recording as a JSON file and get the machine-generated prediction within a couple of seconds.

## How we built it
![CNN model](https://github.com/shehanmunasinghe/diagno/blob/master/Docs/images/3.png?raw=true)

The neural network model used is a 1D-CNN, largely inspired by ResNet.The model takes input as 12 1-dimensional signals corresponding to 12 ECG leads, sampled at 400Hz, and of 12 samples length. At the final layer, the model outputs probabilities for each cardiac condition.

![Workflow](https://github.com/shehanmunasinghe/diagno/blob/master/Docs/images/4.png?raw=true)

The deep neural network model was trained on a subset of 2020 PhysioNet Computing in Cardiology Challenge Data. The trained model is able to predict 5 different cardiac conditions with over 90% accuracy.

Model is deployed on AWS using TorchServe



## What we learned
We got hands-on experience on how to use PyTorch from model building to training to deployment.

## What's next for Diagno
![Raspberry Pi Device](https://github.com/shehanmunasinghe/diagno/blob/master/Docs/images/next_steps.jpeg?raw=true)

As the next step of Diagno, we are planning to build an embedded 12-lead ECG capturing hardware device with a Raspberry Pi and Texas Instruments ADS129X Analog Front End Board