import torch
import torch.nn as nn

class ShallowConvNet_ch(nn.Module):
    def __init__(self, num_classes, input_ch, input_time):
        super(ShallowConvNet_ch, self).__init__()
        self.n_classes = num_classes
        freq = input_time #################### frequency

        self.convnet = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, freq//2), stride=1, bias=False, padding=(1 , freq//4)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(input_ch, 1), stride=1, groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.3),
            nn.Conv2d(16, 16 , kernel_size=(1,freq//4),padding=(0,freq//8), groups=16),
            nn.Conv2d(16, 16, kernel_size=(1,1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.3),
            )
    
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3] # nn.Flatten 역할

        self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes), nn.Dropout(p=0.2))  ####################### classifier 
        
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output=self.clf(output) 
        return output

class DeepConvNet_ch(nn.Module):
    def __init__(self, n_classes,input_ch,input_time, batch_norm=True, batch_norm_alpha=0.1):
        super(DeepConvNet_ch, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 25
        n_ch2 = 50
        n_ch3 = 100
        self.n_ch4 = 200
        
        if self.batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(self.n_ch4, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                )
            
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        self.n_outputs = out.size()[1]*out.size()[2]*out.size()[3]
        self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes), nn.Dropout(p=0.2))  ####################### classifier 

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # output = self.l2normalize(output)
        output=self.clf(output) 

        return output

if __name__ == '__main__':
    model = ShallowConvNet_ch(4, 22, 500) # CLASS, CHANNEL, TIMEWINDOW
    # pred = model(torch.zeros(50, 1, 20, 250))
    print(model)
    from pytorch_model_summary import summary

    print(summary(model, torch.zeros((1, 1, 22, 500)), show_input=True))
            # (1, 1, channel, timewindow)
    
    model = DeepConvNet_ch(4, 22, 500) # CLASS, CHANNEL, TIMEWINDOW
    # pred = model(torch.zeros(50, 1, 20, 250))
    print(model)
    from pytorch_model_summary import summary

    print(summary(model, torch.zeros((1, 1, 22, 500)), show_input=False))
            # (1, 1, channel, timewindow)

