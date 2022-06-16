import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        # Define model
        self.wn1 = nn.utils.weight_norm(nn.Linear(latent_size + 3, 512))
        self.wn2 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.wn3 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.wn4 = nn.utils.weight_norm(nn.Linear(512, 512 - latent_size - 3))

        self.wn5 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.wn6 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.wn7 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.wn8 = nn.utils.weight_norm(nn.Linear(512, 512))

        self.fc = nn.Linear(512, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # implement forward pass
        x = self.dropout(self.relu(self.wn1(x_in)))
        x = self.dropout(self.relu(self.wn2(x)))
        x = self.dropout(self.relu(self.wn3(x)))
        x = self.dropout(self.relu(self.wn4(x)))
        
        x = torch.cat((x, x_in), dim=1)
        x = self.dropout(self.relu(self.wn5(x)))
        x = self.dropout(self.relu(self.wn6(x)))
        x = self.dropout(self.relu(self.wn7(x)))
        x = self.dropout(self.relu(self.wn8(x)))
        
        x = self.fc(x)
        return x
