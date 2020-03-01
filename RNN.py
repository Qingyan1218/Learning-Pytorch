import torch
import torchvision
import os
import matplotlib.pyplot as plt
import time
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data import DataLoader

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(
    mean=[0.5],std=[0.5])])

train_dataset=datasets.MNIST(root='./data',train=True,transform=transform)
test_dataset=datasets.MNIST(root='./data',train=False,transform=transform)
train_load=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_load=DataLoader(test_dataset,batch_size=64,shuffle=False)

images,label=next(iter(train_load))
images_example=torchvision.utils.make_grid(images)
images_example=images_example.numpy().transpose(1,2,0)

mean=[0.5]
std=[0.5]
images_example=images_example*std+mean
plt.imshow(images_example)
plt.show()

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=torch.nn.RNN(input_size=28,
                              hidden_size=128,
                              num_layers=1,
                              batch_first=True)
        self.output=torch.nn.Linear(128,10)

    def forward(self,input):
        output,_=self.rnn(input,None)
        output=self.output(output[:,-1,:])
        return output

model=RNN()
opitimizer=torch.optim.Adam(model.parameters())
loss_f=torch.nn.CrossEntropyLoss()

epoch_n=10
for epoch in range(epoch_n):
    running_loss=0.0
    running_correct=0
    testing_correct=0
    print('Epoch {}/{}'.format(epoch,epoch_n-1))
    print('-'*10)

    for data in train_load:
        X_train,y_train=data
        X_train=X_train.view(-1,28,28)
        X_train,y_train=Variable(X_train),Variable(y_train)
        y_pred=model(X_train)
        loss=loss_f(y_pred,y_train)
        _,pred=torch.max(y_pred.data,1)

        opitimizer.zero_grad()
        loss.backward()
        opitimizer.step()

        running_loss += loss.data
        running_correct += torch.sum(pred==y_train.data)

    for data in test_load:
        X_test,y_test=data
        X_test=X_test.view(-1,28,28)
        X_test,y_test=Variable(X_test),Variable(y_test)
        outputs=model(X_test)
        _,pred=torch.max(outputs.data,1)
        testing_correct +=torch.sum(pred==y_test.data)

    print("Loss is :{:.4f}, Train Accuracy is :{:.4f}%,Test Accuracy is :{:.4f}%".format(
        running_loss/len(train_dataset),100*running_correct/len(train_dataset),
        100*testing_correct/len(train_dataset)))
