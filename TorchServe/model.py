import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
