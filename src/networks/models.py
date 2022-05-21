import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Encoder
class Encoder(nn.Module):
    def __init__(self, opt):

        super(Encoder, self).__init__()
        layer_sizes = opt.encoder_layer_sizes
        latent_size = opt.latent_size
        layer_sizes[0] += latent_size
        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[-1])
        self.fc3 = nn.Linear(layer_sizes[-1], latent_size * 2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(latent_size * 2, latent_size)
        self.linear_log_var = nn.Linear(latent_size * 2, latent_size)
        self.apply(weights_init)

    def forward(self, x, c=None):
        if c is not None:
            x = torch.cat((x, c), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


# Decoder/Generator
class Generator(nn.Module):
    def __init__(self, opt):

        super(Generator, self).__init__()

        layer_sizes = opt.decoder_layer_sizes
        latent_size = opt.latent_size
        input_size = latent_size * 2
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def _forward(self, z, c=None):
        z = torch.cat((z, c), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.sigmoid(self.fc3(x1))
        self.out = x1
        return x

    def forward(self, z, a1=None, c=None, feedback_layers=None):
        if feedback_layers is None:
            return self._forward(z, c)
        else:
            z = torch.cat((z, c), dim=-1)
            x1 = self.lrelu(self.fc1(z))
            feedback_out = x1 + a1 * feedback_layers
            x = self.sigmoid(self.fc3(feedback_out))
            return x


class Discriminator_D1(nn.Module):
    def __init__(self, opt):
        super(Discriminator_D1, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h


class Discriminator_D2(nn.Module):
    def __init__(self, opt):
        super(Discriminator_D2, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        # self.fc3 = nn.Linear(2048, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        h = self.fc2(h)
        return h


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU()
        self.cls = nn.Linear(4096, nclass)

    def forward(self, x):
        o = self.relu(self.fc(x))
        return self.cls(o)
