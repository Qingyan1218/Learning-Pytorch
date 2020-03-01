import torch
import torchvision
import os
import matplotlib.pyplot as plt
import time
from torchvision import datasets, transforms, models
from torch.autograd import Variable

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

model_1=models.vgg16(pretrained=True)
model_2=models.resnet50(pretrained=True)

for param in model_1.parameters():
    param.requires_grad=False

model_1.classifier = torch.nn.Sequential(
    torch.nn.Linear(25088,4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096,4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096,2))

for param in model_2.parameters():
    param.requires_grad=False

model_2.fc=torch.nn.Linear(2048,2)

loss_f_1=torch.nn.CrossEntropyLoss()
loss_f_2=torch.nn.CrossEntropyLoss()

optimizer_1=torch.optim.Adam(model_1.classifier.parameters(),lr=0.00001)
optimizer_2=torch.optim.Adam(model_2.fc.parameters(),lr=0.00001)

weight_1=0.6
weight_2=0.4

epoch_n=2
time_open=time.time()

for epoch in range(epoch_n):
    print('Epoch {}/{}'.format(epoch,epoch_n-1))
    print('-'*10)

    for phase in ['train','valid']:
        if phase=='train':
            print('training...')
            model_1.train(True)
            model_2.train(True)
        else:
            print('validing...')
            model_1.train(False)
            model_2.train(False)
        running_loss_1=0.0
        running_loss_2 = 0.0
        running_corrects_1=0.0
        running_corrects_2 = 0.0
        blending_running_corrects=0

        for batch,data in enumerate(dataloader[phase],1):
            X,y=data
            # X,y=Variable(X).cuda(),Variable(y).cuda()
            X, y = Variable(X), Variable(y)
            y_pred_1=model_1(X)
            y_pred_2=model_2(X)
            blending_y_pred=y_pred_1*weight_1+y_pred_2*weight_2

            _,pred_1=torch.max(y_pred_1.data,1)
            _, pred_2 = torch.max(y_pred_2.data, 1)
            _,blending_pred=torch.max(blending_y_pred.data,1)
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            loss_1=loss_f_1(y_pred_1,y)
            loss_2 = loss_f_2(y_pred_2, y)
            if phase=='train':
                loss_1.backward()
                loss_2.backward()
                optimizer_1.step()
                optimizer_2.step()
            running_loss_1 += loss_1.data
            running_corrects_1 += torch.sum(pred_1==y.data)
            running_loss_2 += loss_2.data
            running_corrects_2 += torch.sum(pred_2==y.data)
            blending_running_corrects += torch.sum(blending_pred==y.data)
            if batch%500==0 and phase=='train':
                print('Batch{},model1 Train Loss:{:.4f},Train ACC:{:.4f}'.format(
                    batch,running_loss_1/batch,100*running_corrects_1/(16*batch)
                ))
                print('Batch{},model2 Train Loss:{:.4f},Train ACC:{:.4f}'.format(
                    batch, running_loss_2 / batch, 100 * running_corrects_2 / (16 * batch)
                ))
        epoch_loss_1=running_loss_1*16/len(image_datasets[phase])
        epoch_loss_2=running_loss_2*16/len(image_datasets[phase])
        epoch_acc_1=100*running_corrects_1/len(image_datasets[phase])
        epoch_acc_2=100*running_corrects_2/len(image_datasets[phase])
        epoch_blending_acc=100*blending_running_corrects/len(image_datasets[phase])
        print("{} model1 Loss:{:.4f} Acc:{:.4f}%".format(phase,epoch_loss_1,epoch_acc_1))
        print("{} model2 Loss:{:.4f} Acc:{:.4f}%".format(phase,epoch_loss_2,epoch_acc_2))
        print('Blending_model Acc{:.4f}'.format(epoch_blending_acc))
    time_end=time.time()-time_open
    print(time_end)
