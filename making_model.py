# Sets device to be GPU if it is available, and CPU if it is not. 
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
  
 # Prints chosen device
print(dev)

# Creates model architecture 
class simple_net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input: 3x32x32
        

        # Defines average pooling layers to carry out spatial average pooling. 
        # The layer adapts to the input size of the feature map and pools it down 
        # to a 1x1 value. 
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # Defines max pooling layer with a kernel size of 2x2.
        self.max_pool = nn.MaxPool2d(2,2)
        
        # Defines average pooling layer with a kernel size of 2x2
        self.avg_pool = nn.AvgPool2d(2,2)
        
        # Defines linear layers which which are used to create vector a. 
        self.linear = nn.Linear(3 * 1 * 1, 3)
        self.linear2 = nn.Linear(120 * 1 * 1, 3)
        self.linear3 = nn.Linear(240 * 1 * 1, 3)

        # Defines final output linear layer. 
        self.linear_output = nn.Linear(360*1*1,10)
        
        # Defines three convolutional layers, which will make up three blocks within the network. 
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 120, kernel_size = 5, padding = 2)
        self.conv2 = nn.Conv2d(in_channels = 120, out_channels = 240, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 240, out_channels = 360, kernel_size = 3, padding = 1)

        # Defines Batch Normalization 
        # Batch normalizations are used to stabilize the training process. 
        self.batch1 = nn.BatchNorm2d(120)
        self.batch2 = nn.BatchNorm2d(240)
        self.batch3 = nn.BatchNorm2d(360)
    
    def forward(self, x):
        
        ### BLOCK ONE ###
        
        # a is generated. 
        a = F.relu(self.linear(torch.flatten(self.pool(x),1)))
        # three convolutional layers recieve the x input, after which a batch normalisation, an activation function 
        # and pooling layer are applied. 
        conv_output_1 = self.max_pool(F.relu(self.batch1(self.conv1(x))))
        conv_output_2 = self.max_pool(F.relu(self.batch1(self.conv1(x))))
        conv_output_3 = self.max_pool(F.relu(self.batch1(self.conv1(x))))
        combined_conv = []
        # Each convolutional layer output is multiplied by its respective a value. The for loop goes through each element 
        # of the batch. 
        for i in range(len(a)):
            combined_conv.append(conv_output_1[i] * a[i][0] + conv_output_2[i] * a[i][1] + conv_output_3[i] * a[i][2])
        # The stack function is then used to combine the results back into a tensor of dimensions: (batch-size, channel numbers, height, width)
        O = torch.stack(combined_conv)
        
        ### BLOCK TWO ###
        a = F.relu(self.linear2(torch.flatten(self.pool(O),1)))
        conv_output_1 = self.max_pool(F.relu(self.batch2(self.conv2(O))))
        conv_output_2 = self.max_pool(F.relu(self.batch2(self.conv2(O))))
        conv_output_3 = self.max_pool(F.relu(self.batch2(self.conv2(O))))
        combined_conv = []
        for i in range(len(a)):
            combined_conv.append(conv_output_1[i] * a[i][0] + conv_output_2[i] * a[i][1] + conv_output_3[i] * a[i][2])
        O_2 = torch.stack(combined_conv)
        
        ### BLOCK THREE ###
        a = F.relu(self.linear3(torch.flatten(self.pool(O_2),1)))
        conv_output_1 = F.relu(self.batch3(self.conv3(O_2)))
        conv_output_2 = F.relu(self.batch3(self.conv3(O_2)))
        conv_output_3 = F.relu(self.batch3(self.conv3(O_2)))
        combined_conv = []
        for i in range(len(a)):
            combined_conv.append(conv_output_1[i] * a[i][0] + conv_output_2[i] * a[i][1] + conv_output_3[i] * a[i][2])
        O_3 = torch.stack(combined_conv)
        
        # Classifier 
        # Spatial average pooling is applied to the output of the last block
        x = self.pool(O_3)
        # The tensor is flattened into a flat vector. 
        x = torch.flatten(x, 1)
        # The final classifier is a linear layer with a relu activation function. 
        x = F.relu(self.linear_output(x))
        return x

# Network is edefined and sent to the active device

simple_net = simple_net().to(dev)

# Importing optimizer
import torch.optim as optim
# Importing optimizer schedueler
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Learning rate of 0.0001 is defined. 
lr = 0.001

# CrossEntropyLoss is used to generate a loss value for the network.
criterion = nn.CrossEntropyLoss()
# Stochastic Gradient Descent with momentum of 0.9 is used as the optimizer
optimizer = optim.SGD(simple_net.parameters(),lr=lr, momentum = 0.9)
# A learning rate scheduler is defined to lower the learning rate if the validation accuracy is 
# not improved after 6 epochs. 
scheduler = ReduceLROnPlateau(optimizer, 'max', patience = 6, verbose = True)
