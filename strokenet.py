import math
import cv2
import matplotlib
from matplotlib import pyplot as plt
from torch import Tensor, nn
import torch
from torchvision import models


class FE_HeadModule(nn.Module):
    def __init__(self):
        super(FE_HeadModule, self).__init__()
        n_filters = [1, 64, 128, 256, 256, 512]
        self.conv_layers = nn.ModuleList()
        for i in range(len(n_filters) - 1):
            self.conv_layers.append(
                nn.Conv2d(n_filters[i], n_filters[i + 1], 3, padding=1))
            self.conv_layers.append(nn.Dropout(0.5))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(
                nn.Conv2d(n_filters[i+1], n_filters[i + 1], 3, padding=1))
            self.conv_layers.append(nn.Dropout(0.5))
            self.conv_layers.append(nn.LeakyReLU())
            self.conv_layers.append(nn.MaxPool2d(2))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x  # 256x256 -> 8x8


class FE_TailModule(nn.Module):
    def __init__(self):
        super(FE_TailModule, self).__init__()
        n_filters = [512, 512, 512, 512]
        self.conv_layers = nn.ModuleList()
        for i in range(len(n_filters) - 1):
            self.conv_layers.append(
                nn.Conv2d(n_filters[i], n_filters[i+1], 3, padding=1))
            self.conv_layers.append(nn.Dropout(0.5))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(
                nn.Conv2d(n_filters[i+1], n_filters[i+1], 3, padding=1))
            self.conv_layers.append(nn.Dropout(0.5))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(2))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x  # 16x16 -> 2x2x512


class VGGBasedEncoder(nn.Module):
    def __init__(self):
        super(VGGBasedEncoder, self).__init__()
        self.inp_adjust = nn.Sequential(
            nn.Conv2d(1, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.encoder = models.vgg16_bn(weights=None)
        self.encoder.classifier = nn.Sequential(
            nn.Tanh(),
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, 256, 256))
        # resize to 224x224
        x = nn.functional.interpolate(x, size=224)
        x = self.inp_adjust(x)
        x = self.encoder(x)
        return x


class StrokeSingleAttn(nn.Module):
    def __init__(self):
        super(StrokeSingleAttn, self).__init__()
        self.full_img_encoder = FE_HeadModule()
        self.stroke_encoder = FE_HeadModule()
        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)
        self.query = nn.Conv2d(512, 512, 3, padding=1)
        self.key = nn.Conv2d(512, 512, 3, padding=1)
        self.value = nn.Conv2d(512, 512, 3, padding=1)

        self.out = nn.Sequential(
            FE_TailModule(),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
        )

    def forward(self, x, full_img, return_attn=False):
        x = torch.reshape(x, (-1, 1, 256, 256))
        full_img = torch.reshape(full_img, (-1, 1, 256, 256))

        x = self.stroke_encoder(x)
        f = self.full_img_encoder(full_img)
        # divide 4096 into 16 time steps of 512
        q = self.query(x)
        k = self.key(x)
        v = x
        q = q.reshape(-1, 8*8, 512)
        k = k.reshape(-1, 8*8, 512)
        v = v.reshape(-1, 8*8, 512)

        attn_output, attn_output_weights = self.attn(q, k, v)
        attn_output = attn_output.reshape(-1, 512, 8, 8)
        x = attn_output
        x = self.out(x) + torch.tensor([[0.8327963,  0.05671397, 0.04100104,
                                         0.02160976, 0.8883327,  0.0242271]]).to(x.device)
        # print(x.shape)
        if return_attn:
            return x, attn_output_weights
        return x


class StrokeSingleVGG(nn.Module):
    def __init__(self, img_size=256):
        super(StrokeSingleVGG, self).__init__()
        self.inp_adjust = nn.Sequential(
            nn.Conv2d(1, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        # self.res18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vgg = models.vgg16_bn(weights=None)
        self.vgg.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 6),
            # nn.Tanh(),
        )

    def forward(self, x, full_img=None):
        x = torch.reshape(x, (-1, 1, 256, 256))
        # full_img = torch.reshape(full_img, (-1, 1, 256, 256))
        # resize to 224x224
        # full_img = nn.functional.interpolate(full_img, size=224)
        x = nn.functional.interpolate(x, size=224)

        # concat full_img and x
        # x = torch.cat((full_img, x), dim=1)
        x = self.inp_adjust(x)
        x = self.vgg(x)
        return x


class StrokeSingleRes18(nn.Module):
    def __init__(self, img_size=256):
        super(StrokeSingleRes18, self).__init__()
        self.inp_adjust = nn.Sequential(
            nn.Conv2d(2, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        # self.res18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.res18 = models.resnet18(weights=None)
        self.res18.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 6),
            # nn.Tanh(),
        )

    def forward(self, x, full_img=None):
        x = torch.reshape(x, (-1, 1, 256, 256))
        full_img = torch.reshape(full_img, (-1, 1, 256, 256))
        # concat full_img and x
        x = torch.cat((full_img, x), dim=1)
        x = self.inp_adjust(x)
        x = self.res18(x)
        return x


class StrokeSingleNet(nn.Module):
    def __init__(self, image_size=256):
        super(StrokeSingleNet, self).__init__()
        n_filters = [1, 64, 128, 256, 256, 512, 512, 512]
        self.conv_layers = nn.ModuleList()
        for i in range(len(n_filters) - 1):
            self.conv_layers.append(
                nn.Conv2d(n_filters[i], n_filters[i + 1], 3, padding=1))
            self.conv_layers.append(nn.BatchNorm2d(n_filters[i + 1]))
            self.conv_layers.append(nn.Dropout(0.5))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(2))

        n_feats = n_filters[-1] * (image_size // (2 ** (len(n_filters) - 1))) ** 2
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(n_feats, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.output = nn.Linear(1024, 6)

    def forward(self, x, full_img=None):
        # full_img = torch.reshape(full_img, (-1, 1, 256, 256))
        x = torch.reshape(x, (-1, 1, 256, 256))
        # # concat full_img and x
        # x = torch.cat((full_img, x), dim=1)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        x = self.output(x)
        return x


class DNN(nn.Module):
    def __init__(self, *args):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(7*64*64, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            # nn.Linear(4096, 4096),
            # nn.Dropout(0.5),
            # nn.ReLU(),
            nn.Linear(4096, 6*6),
        )

    def forward(self, full_img, stroke_img):
        stroke_img = torch.reshape(stroke_img, (-1, 6, 256, 256))
        # resize full_img to 64x64
        full_img = nn.functional.interpolate(full_img, size=64)
        # resize stroke_img to 64x64
        stroke_img = nn.functional.interpolate(stroke_img, size=64)
        # stack full_img and stroke_img
        # print(full_img.shape, stroke_img.shape)
        stroke_img = torch.reshape(stroke_img, (-1, 6*64*64))
        full_img = torch.reshape(full_img, (-1, 1*64*64))
        x = torch.cat((full_img, stroke_img), dim=1)
        x = self.layers(x)
        x = torch.reshape(x, (-1, 6, 6))
        return x


class SimpleModel(nn.Module):
    def __init__(self, *args):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(7, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 6*6),
        )

    def forward(self, full_img, stroke_img):
        # stack full_img and stroke_img
        stroke_img = torch.reshape(stroke_img, (-1, 6, 256, 256))
        full_img = torch.reshape(full_img, (-1, 1, 256, 256))
        x = torch.cat((full_img, stroke_img), dim=1)
        x = self.layers(x)
        x = torch.reshape(x, (-1, 6, 6))
        return x


class StrokeVGGEncoder(nn.Module):
    def __init__(self, dim=512):
        super(StrokeVGGEncoder, self).__init__()
        self.stroke_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stroke_encoder.fc = nn.Sequential(
            nn.Linear(512, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # make it able to be finetuned
        for param in self.stroke_encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        # repeat channel from 1 to 3
        x = x.repeat(1, 3, 1, 1)
        x = self.stroke_encoder(x)
        return x


class StrokeTransformer(nn.Module):
    def __init__(self, dim=1024):
        super(StrokeTransformer, self).__init__()
        self.dim = dim
        self.stroke_encoder = StrokeVGGEncoder(dim=dim)
        self.transformer = nn.Transformer(d_model=dim, nhead=8, num_encoder_layers=12,
                                          num_decoder_layers=12, dim_feedforward=4096,
                                          dropout=0.1, activation='gelu', batch_first=True)
        # trans 6d to dim
        self.decoder_trans = nn.Sequential(
            nn.Linear(6, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.final_decoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, 6),
        )

        self.tanh = nn.Tanh()

    def forward(self, full_img, stroke_img, output, mask=None):
        """
        stroke_img: [batch_size, seq_len, 1, 256, 256]
        full_img: [batch_size, 3, 256, 256]
        decoder_inp: [batch_size, seq_len, 6]
        """
        # img_feature needs to be iterated over time step
        img_feature = torch.zeros(
            (stroke_img.shape[0], stroke_img.shape[1], self.dim)).to(
            stroke_img.device)
        for i in range(stroke_img.shape[1]):
            img_feature[:, i] = self.stroke_encoder(stroke_img[:, i])
        # img_feature: [batch_size, seq_len, dim]
        # use encoding of full_img as decoder BOS
        full_img_feature = self.stroke_encoder(full_img)
        # decoder_inp: [batch_size, seq_len, 6]
        # tgt_mask needs to be generated for sequentially generating strokes
        # decode from output to decoder_inp
        tgt_inp = torch.zeros((output.shape[0], output.shape[1], img_feature.shape[-1])).to(
            output.device)

        for i in range(0, output.shape[1]):
            tgt_inp[:, i] = self.decoder_trans(output[:, i])

        tgt_inp = torch.cat((full_img_feature.unsqueeze(1), tgt_inp), dim=1)[:, :-1, :]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_inp.shape[1]).to(
            tgt_inp.device)

        feats = self.transformer(img_feature, tgt_inp,
                                 #  src_key_padding_mask=torch.logical_not(mask),
                                 #  tgt_key_padding_mask=torch.logical_not(mask),
                                 tgt_mask=tgt_mask)

        # feats: [batch_size, seq_len, dim]
        # final_feats: [batch_size, seq_len, 6]
        final_feats = self.final_decoder(feats)
        # final_feats = self.tanh(final_feats) * 3
        return final_feats


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.Dropout(0.5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = x + self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        return x


class CharImageEncoder(nn.Module):
    def __init__(self, img_size=128, dim=512):
        """
            Encoder for calligraphy character image
            input: square grayscale image
            output: N dim feature vector
        """
        super(CharImageEncoder, self).__init__()
        self.img_size = img_size
        n_filters = [1, 32, 64, 128, 256, 256, 512, 512]
        self.conv_layers = nn.ModuleList()
        for i in range(len(n_filters) - 1):
            # self.conv_layers.append(
            #     nn.Conv2d(n_filters[i], n_filters[i + 1], 3, padding=1))
            # self.conv_layers.append(nn.BatchNorm2d(n_filters[i + 1]))
            # self.conv_layers.append(nn.Dropout(0.5))
            # self.conv_layers.append(nn.ReLU())
            # self.conv_layers.append(nn.MaxPool2d(2))
            self.conv_layers.append(ResBlock(n_filters[i], n_filters[i + 1]))
        final_size = img_size // (2 ** (len(n_filters) - 1))
        print("final dim:", final_size * final_size * n_filters[-1])
        self.fc = nn.Linear(n_filters[-1] * final_size * final_size, dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        # resize to self.img_size
        x = nn.functional.interpolate(x, size=self.img_size)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.activation(x)
        return x


class StrokeImageEncoder(nn.Module):
    def __init__(self, img_size=128, dim=256):
        """
            Encoder for a single stroke image. stroke should be centered zero padded.
            input: square grayscale image
            output: N dim feature vector
        """
        super(StrokeImageEncoder, self).__init__()
        self.img_size = img_size
        n_filters = [1, 32, 64, 128, 256, 256, 512, 512]
        self.conv_layers = nn.ModuleList()
        for i in range(len(n_filters) - 1):
            # self.conv_layers.append(
            #     nn.Conv2d(n_filters[i], n_filters[i + 1], 3, padding=1))
            # self.conv_layers.append(nn.BatchNorm2d(n_filters[i + 1]))
            # self.conv_layers.append(nn.Dropout(0.5))
            # self.conv_layers.append(nn.ReLU())
            # self.conv_layers.append(nn.MaxPool2d(2))
            self.conv_layers.append(ResBlock(n_filters[i], n_filters[i + 1]))
        final_size = img_size // (2 ** (len(n_filters) - 1))
        self.fc = nn.Linear(n_filters[-1] * final_size * final_size, dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        # resize to self.img_size
        x = nn.functional.interpolate(x, size=self.img_size)

        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.activation(x)
        return x


class StrokeNet(nn.Module):
    def __init__(self, hidden_dim=512):
        super(StrokeNet, self).__init__()
        self.char_encoder = CharImageEncoder(dim=hidden_dim)
        # stroke_encoder is time-distributed
        # self.pe = PositionalEncoding(hidden_dim)
        self.strokes_encoder = TimeDistributed(
            StrokeImageEncoder(dim=hidden_dim),
            batch_first=True
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=0.5, num_layers=4)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
        #                                                 nhead=1,
        #                                                 batch_first=True,
        #                                                 dropout=0.5,
        #                                                 activation='relu')
        # self.attn = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.output = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim // 2, 6, kernel_size=1)
        )

        self.tanh = nn.Tanh()

    def forward(self, char_img, strokes_img, mask=None):
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
        # feature = self.pe(feature)  # [batch_size, seq_len, hidden_dim]
        # hidden = self.attn(feature, src_key_padding_mask=~mask)  # [batch_size, seq_len, hidden_dim]

        hidden, _ = self.rnn(feature)  # [batch_size, seq_len, hidden_dim]
        # output, attn_map = self.attn(feature.transpose(0, 1), feature.transpose(0, 1),
        #                     feature.transpose(0, 1), )  # [seq_len, batch_size, hidden_dim]

        output = self.output(hidden.transpose(1, 2))  # [batch_size, 6, seq_len]
        output = output.transpose(1, 2)  # [batch_size, seq_len, 6]
        output = self.tanh(output) * 3
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
        model = StrokeTransformer().to('cuda' if torch.cuda.is_available() else 'cpu')
        stroke_img = torch.randn((2, 10, 1, 256, 256)).to('cuda' if torch.cuda.is_available() else 'cpu')
        full_img = torch.randn((2, 1, 256, 256)).to('cuda' if torch.cuda.is_available() else 'cpu')
        decoder_inp = torch.randn((2, 10, 6)).to('cuda' if torch.cuda.is_available() else 'cpu')
        mask = torch.ones((2, 10), dtype=torch.bool).to('cuda' if torch.cuda.is_available() else 'cpu')
        output = model(full_img, stroke_img, decoder_inp, mask)
        print(output.shape)
        output.mean().backward()

    my_test()
