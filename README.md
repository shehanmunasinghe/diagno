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


    class ECGNet(nn.Module):
        
        def __init__(self, input_channels=12, N_labels=2, kernel_size =17,  n_blocks=4):
            super().__init__()

            self.padding= (kernel_size-1)//2
            self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=kernel_size, padding=self.padding) # input_channelsx4096 -> #64x4096
            self.bn1 = nn.BatchNorm1d(64) #64x4096
            self.relu1 = nn.ReLU() #64x4096

            self.resblock1 = self.ResBlock(64,4096,128,1024)
            self.resblock2 = self.ResBlock(128,1024,196,256)
            self.resblock3 = self.ResBlock(196,256, 256, 64)
            self.resblock4 = self.ResBlock(256,64, 320, 16)
            
            self.flatten = nn.Flatten()
            self.dense_final = nn.Linear(320*16, N_labels)
            self.sigmoid_final = nn.Sigmoid()
        
        def forward(self, x_in):
            
            x = self.conv1(x_in)
            x = self.bn1(x)
            x = self.relu1(x)

            x, y = self.resblock1((x,x))
            x, y = self.resblock2((x,y))
            x, y = self.resblock3((x,y))
            x, _ = self.resblock4((x,y))
            
            x = self.flatten(x)
            x = self.dense_final(x)
            x = self.sigmoid_final(x)

            return x 
    
        class ResBlock(nn.Module):
        def __init__(self, n_filters_in, n_samples_in, n_filters_out, n_samples_out,
                    dropout_rate=0.8, kernel_size=17):
            super(ECGNet.ResBlock, self).__init__()
            self.padding=(kernel_size-1)//2 
            downsample= n_samples_in//n_samples_out 

            self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size=kernel_size, padding=self.padding) 
            self.bn1 = nn.BatchNorm1d(n_filters_out)
            self.relu1 = nn.ReLU() 
            self.dropout1 = nn.Dropout(p=dropout_rate)
            self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size=kernel_size, stride=downsample, padding=self.padding) 
            
            self.sk_max_pool= nn.MaxPool1d(downsample)
            self.sk_conv = nn.Conv1d(n_filters_in, n_filters_out, kernel_size=1) 
            
            self.bn2 = nn.BatchNorm1d(n_filters_out) 
            self.relu2 = nn.ReLU() 
            self.dropout2 = nn.Dropout(p=dropout_rate)
            
            

        def forward(self, inputs):
            x,y = inputs
            y = self.sk_max_pool(y)# skip connection (Max Pool -> 1dConv)
            y = self.sk_conv(y)
            x = self.conv1(x) #Conv1d
            x = self.bn1(x) #bn
            x = self.relu1(x) #ReLU
            x = self.dropout1(x) #dropout 
            x = self.conv2(x) #conv
            x = x+y
            y = x
            x = self.bn2(x) #bn
            x = self.relu2(x) #relu
            x = self.dropout2(x) #dropout 
            return x,y


![Workflow](https://github.com/shehanmunasinghe/diagno/blob/master/Docs/images/4.png?raw=true)

The deep neural network model was trained on a subset of 2020 PhysioNet Computing in Cardiology Challenge Data. The trained model is able to predict 5 different cardiac conditions with over 90% accuracy.

Model is deployed on AWS using TorchServe



## What we learned
We got hands-on experience on how to use PyTorch from model building to training to deployment.

## What's next for Diagno
![Raspberry Pi Device](https://github.com/shehanmunasinghe/diagno/blob/master/Docs/images/next_steps.jpeg?raw=true)

As the next step of Diagno, we are planning to build an embedded 12-lead ECG capturing hardware device with a Raspberry Pi and Texas Instruments ADS129X Analog Front End Board