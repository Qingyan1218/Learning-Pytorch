import os

import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image

im_tfs = tfs.Compose([
    tfs.ToTensor(),
    # tfs.Lambda(lambda x: x.repeat(3,1,1)),
    # tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])# standardize
    tfs.Normalize([0.5],[0.5])])
    # the gray picture has one channel

train_set = MNIST(root='./data', transform=im_tfs)
train_data = DataLoader(train_set, batch_size=128, shuffle=True)

# define the net
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)
        ) # output 3 dimension to show the picture

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

net = autoencoder()
# x = Variable(torch.randn(1, 28*28))
# #generate a sample to test the net
# code, _ = net(x)
# print(code.shape)

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

def to_img(x):
    '''
    define a funciton to change the result to picture
    '''
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    # put che data into the range(0,1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x


# begin to train autoencode
for e in range(2):
    for im, _ in train_data:
        # im=im[:,0,:,:]
        im = Variable(im.view(im.shape[0],im.shape[2]*im.shape[3]))
        # print(im.shape)
        # forward
        _, output = net(im)
        loss = criterion(output, im) / im.shape[0]  # average
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (e + 1) % 2 == 0:  # every 2 times save the pictures
        print('epoch: {}, Loss: {:.4f}'.format(e + 1, loss.data))
        pic = to_img(output.cpu().data)
        if not os.path.exists('./simple_autoencoder'):
            os.mkdir('./simple_autoencoder')
        save_image(pic, './simple_autoencoder/image_{}.png'.format(e + 1))

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# #show the result
view_data = Variable((train_set.data[:200].type(torch.FloatTensor).view(-1, 28*28) / 255. - 0.5) / 0.5)
encode, _ = net(view_data)    # get the compressed result
fig = plt.figure(2)
ax = Axes3D(fig)    # 3D
# x, y, z value
X = encode.data[:, 0].numpy()
Y = encode.data[:, 1].numpy()
Z = encode.data[:, 2].numpy()
values = train_set.targets[:200].numpy()  # label value
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9))    # color
    ax.text(x, y, z, s, backgroundcolor=c)  # place
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()


code = Variable(torch.FloatTensor([[1.19, -3.36, 2.06]]))
decode = net.decoder(code)
decode_img = to_img(decode).squeeze()
decode_img = decode_img.data.numpy() * 255
plt.imshow(decode_img.astype('uint8'), cmap='gray')
# generate a picture 3
plt.show()

# use convolutional network to encode and decode
class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (b, 8, 3, 3)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

conv_net = conv_autoencoder()
if torch.cuda.is_available():
    conv_net = conv_net.cuda()
optimizer = torch.optim.Adam(conv_net.parameters(), lr=1e-3, weight_decay=1e-5)

# 开始训练自动编码器
for e in range(40):
    for im, _ in train_data:
        if torch.cuda.is_available():
            im = im.cuda()
        im = Variable(im)
        # 前向传播
        _, output = conv_net(im)
        loss = criterion(output, im) / im.shape[0]  # 平均
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (e + 1) % 20 == 0:  # 每 20 次，将生成的图片保存一下
        print('epoch: {}, Loss: {:.4f}'.format(e + 1, loss.data))
        pic = to_img(output.cpu().data)
        if not os.path.exists('./conv_autoencoder'):
            os.mkdir('./conv_autoencoder')
        save_image(pic, './conv_autoencoder/image_{}.png'.format(e + 1))
