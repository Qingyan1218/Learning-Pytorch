import torch
import torchvision
import os
import matplotlib.pyplot as plt
import time
from torchvision import datasets, transforms, models
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')

data_dir='DogsVSCats_small'
data_transform={x:transforms.Compose([transforms.Resize([224,224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
                for x in ["train","valid"]}

image_datasets={x:datasets.ImageFolder(root=os.path.join(data_dir,x),
                                       transform=data_transform[x])
                for x in ["train","valid"]}

dataloader={x:torch.utils.data.DataLoader(dataset=image_datasets[x],
                                          batch_size=16,
                                          shuffle=True)
            for x in ["train","valid"]}

index_classes=image_datasets["train"].class_to_idx
example_classes=image_datasets['train'].classes

model=models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad=False

model.fc=torch.nn.Linear(2048,2)

# model=model.cuda()
# GPU memory is not enough

loss_f=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.fc.parameters(),lr=0.00001)

epoch_n=2
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
            # X,y=Variable(X).cuda(),Variable(y).cuda()
            X, y = Variable(X), Variable(y)
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
