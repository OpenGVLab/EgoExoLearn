import torch
from torch import nn

class RAAN(nn.Module):
    def __init__(self, num_features, attention, num_filters=3, input_size=1024, fc_output=256):
        super(RAAN, self).__init__()
        self.num_features = num_features
        self.attention = attention
        self.num_filters = num_filters
        self.input_size = input_size
        self.fc_output = fc_output

        self._prepare_raan_model() # att_net and fc

    def _prepare_raan_model(self):
        self.att_net = nn.ModuleList()

        for i in range(0, self.num_filters):
            self.att_net.append(nn.Sequential(
                nn.Linear(self.input_size, self.fc_output),
                nn.ReLU(),
                nn.Linear(self.fc_output, 1),
                nn.Softmax(dim=1)))

        self.fc = nn.Sequential(
            nn.Linear(self.input_size, 1),
            nn.Tanh())

    def forward(self, input):
        # print(input.shape) # 128 400 1024
        input = input.view(-1, self.num_features, self.input_size)
        # print(input.shape) # 128 400 1024
        # print(input.shape) # 128 10 1024
        # exit()
        if self.attention:
            att_list = []
            for i in range(0, self.num_filters):
                att_list.append(self.att_net[i](input))
            all_atts = torch.stack(att_list, 2)
            # print('att')
        else:
            all_atts = torch.ones((input.size(0),self.num_features, self.num_filters, 1)).cuda() * (1.0/self.num_features)
            # print('no att')
        #att = torch.mean(all_atts, 2)
        outputs = []
        for i in range(0, self.num_filters):
            tmp_outputs = torch.mul(input, all_atts[:,:,i,:])
            tmp_outputs = tmp_outputs.sum(1)

            outputs.append(self.fc(tmp_outputs).view(-1)*2)
        all_outputs = torch.stack(outputs, 1)
        return all_outputs, all_atts, tmp_outputs 
        
        # if there is no attention, tmp_outputs should not be used in triplet loss
        # if there is attention, relation network architecture is better fitted
        # triplet loss and RN should not be used together
