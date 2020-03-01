import torch
import torchvision
import os
import matplotlib.pyplot as plt
import time
from torchvision import datasets, transforms
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')

data_dir='DogsVSCats'
data_transform={x:transforms.Compose([transforms.Resize([64,64]),
                                      transforms.ToTensor()])
                for x in ["train","valid"]}

image_datasets={x:datasets.ImageFolder(root=os.path.join(data_dir,x),
                                       transform=data_transform[x])
                for x in ["train","valid"]}

dataloader={x:torch.utils.data.DataLoader(dataset=image_datasets[x],
                                          batch_size=16,
                                          shuffle=True)
            for x in ["train","valid"]}

X_example,y_example=next(iter(dataloader['train']))
print('X_example is {}'.format(len(X_example)))
print('y_example is {}'.format(len(y_example)))

# X_example.shape is (16,3,64,64)
index_classes=image_datasets["train"].class_to_idx
print(index_classes)

example_classes=image_datasets['train'].classes
print(example_classes)

# img =torchvision.utils.make_grid(X_example)
# img = img.numpy().transpose([1,2,0])
# print([example_classes[i] for i in y_example])
# plt.imshow(img)
# plt.show()

class Models(torch.nn.Module):
    def __init__(self):
        super(Models,self).__init__()
        self.Conv=torch.nn.Sequential(
            torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.Classes=torch.nn.Sequential(
            torch.nn.Linear(4*4*512,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,2)
        )

    def forward(self, input):
        x=self.Conv(input)
        x=x.view(-1,4*4*512)
        x=self.Classes(x)
        return x

model=Models().cuda()
print(model)

loss_f=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.00001)

epoch_n=1
time_open=time.time()

for epoch in range(epoch_n):
    print('Epoch {}/{}'.format(epoch,epoch_n-1))
    print('-'*10)

    for phase in ['train','valid']:
        if phase=='train':
            print('training...')
            model.train(True)
        else:
            print('validing...')
            model.train(False)
        running_loss=0.0
        running_corrects=0.0

        for batch,data in enumerate(dataloader[phase],1):
            X,y=data
            X,y=Variable(X).cuda(),Variable(y).cuda()
            y_pred=model(X)
            _,pred=torch.max(y_pred.data,1)
            optimizer.zero_grad()
            loss=loss_f(y_pred,y)
            if phase=='train':
                loss.backward()
                optimizer.step()
            running_loss += loss.data
            running_corrects += torch.sum(pred==y.data)
            if batch%500==0 and phase=='train':
                print('Batch{},Train Loss:{:.4f},Train ACC:{:.4f}'.format(
                    batch,running_loss/batch,100*running_corrects/(16*batch)
                ))
        epoch_loss=running_loss*16/len(image_datasets[phase])
        epoch_acc=100*running_corrects/len(image_datasets[phase])
        print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase,epoch_loss,epoch_acc))
    time_end=time.time()-time_open
    print(time_end)

