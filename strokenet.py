from torch import nn
import torch


class CharImageEncoder(nn.Module):
    def __init__(self, img_size=256, dim=256):
        """
            Encoder for calligraphy character image
            input: square grayscale image
            output: N dim feature vector
        """
        super(CharImageEncoder, self).__init__()
        n_filters = [1, 16, 32, 64, 128, 128, 128]
        self.conv_layers = nn.ModuleList()
        for i in range(len(n_filters) - 1):
            self.conv_layers.append(
                nn.Conv2d(n_filters[i], n_filters[i + 1], 3, padding=1))
            self.conv_layers.append(nn.BatchNorm2d(n_filters[i + 1]))
            self.conv_layers.append(nn.Dropout(0.5))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(2))
        final_size = img_size // (2 ** (len(n_filters) - 1))
        self.fc = nn.Linear(n_filters[-1] * final_size * final_size, dim)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class StrokeImageEncoder(nn.Module):
    def __init__(self, img_size=256, dim=256):
        """
            Encoder for a single stroke image. stroke should be centered zero padded.
            input: square grayscale image
            output: N dim feature vector
        """
        super(StrokeImageEncoder, self).__init__()
        n_filters = [1, 16, 32, 64, 128, 128, 128]
        self.conv_layers = nn.ModuleList()
        for i in range(len(n_filters) - 1):
            self.conv_layers.append(
                nn.Conv2d(n_filters[i], n_filters[i + 1], 3, padding=1))
            self.conv_layers.append(nn.BatchNorm2d(n_filters[i + 1]))
            self.conv_layers.append(nn.Dropout(0.5))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(2))
        final_size = img_size // (2 ** (len(n_filters) - 1))
        self.fc = nn.Linear(n_filters[-1] * final_size * final_size, dim)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class StrokeNet(nn.Module):
    def __init__(self, hidden_dim=256):
        super(StrokeNet, self).__init__()
        self.char_encoder = CharImageEncoder(dim=hidden_dim)
        # stroke_encoder is time-distributed

        self.strokes_encoder = TimeDistributed(
            StrokeImageEncoder(dim=hidden_dim),
            batch_first=True
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Conv1d(hidden_dim, 7, kernel_size=1)

    def forward(self, char_img, strokes_img):
        """
        :param char_img: [batch_size, 1, 256, 256]
        :param strokes_img: [batch_size, seq_len, 1, 256, 256]
        :return: [batch_size, seq_len, 6]
        """
        char_feature = self.char_encoder(char_img)  # [batch_size, hidden_dim]
        strokes_feature = self.strokes_encoder(strokes_img)  # [batch_size, seq_len, hidden_dim]
        # repeat char feature and add to stroke feature
        char_feature = char_feature.repeat(1, strokes_feature.shape[1]).view(
            strokes_feature.shape[0], strokes_feature.shape[1], -1
        )  # [batch_size, seq_len, hidden_dim]

        feature = char_feature + strokes_feature  # [batch_size, seq_len, hidden_dim]
        hidden, _ = self.rnn(feature)  # [batch_size, seq_len, hidden_dim]
        output = self.output(hidden.transpose(1, 2))  # [batch_size, 6, seq_len]
        output = output.transpose(1, 2)  # [batch_size, seq_len, 6]
        return output


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x, *args):
        output = torch.Tensor().to(x.device)
        if self.batch_first:
            for i in range(x.shape[1]):
                output = torch.cat(
                    (output,
                     self.module(x[:, i], *args).view(x.shape[0], 1, -1)), dim=1)
        else:
            raise NotImplementedError
        return output


if __name__ == '__main__':
    def my_test():
        print("TEST!!!")
        model = StrokeNet()
        char_img = torch.randn(32, 1, 256, 256)
        stroke_img = torch.randn(32, 8, 1, 256, 256)
        output = model(char_img, stroke_img)
        print(output.shape)


    my_test()
