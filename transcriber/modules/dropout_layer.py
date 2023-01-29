import torch.nn as nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class dropout_layer(nn.Module):

    def __init__(self):
        super(dropout_layer, self).__init__()
        self.drop = nn.Dropout(p=0.5)

    def forward(self, input):
        
        #a = torch.ones(input.shape, device= device)
        dummy_array_input = torch.ones(input.shape[1], device= device) #For dim 1 (T)
        dummy_array_output = (self.drop(dummy_array_input).to(device)) * (1/1-(0.5)) #Drop 50% # Scaling factor 1/1-p
        dummy_array_output_t = torch.reshape(dummy_array_output, (input.shape[1], 1)).to(device) #Transpose
        dummy_array_output_f = dummy_array_output_t.repeat(input.shape[0], 1,input.shape[2]).to(device) #Same size as input
        
        output =  input*dummy_array_output_f  #element-wise multiplication
        return output