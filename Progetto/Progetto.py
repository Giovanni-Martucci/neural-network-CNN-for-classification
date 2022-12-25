from glob import glob
from os.path import basename
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
import torch
from torch.utils import data
from os.path import join
from PIL import Image
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from torchvision.models import resnet18
from torchvision.models import resnet50
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score
from torchvision.models.googlenet import googlenet
from torchvision.models import squeezenet1_0
import os



class AverageValueMeter():
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0
        self.num = 0
    def add(self, value, num):
        self.sum += value*num
        self.num += num
    def value(self):
        try:
            return self.sum/self.num
        except:
            return None


class CSVImageDataset(data.Dataset):
    def __init__(self, data_root, csv, transform = None):
        self.data_root = data_root
        self.data = pd.read_csv(csv)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        im_path, im_label = self.data.iloc[i]['path'], self.data.iloc[i].label
        #il dataset contiene alcune immagini in scala di grigi
        #convertiamo tutto in RGB per avere delle immagini consistenti
        im = Image.open(join(self.data_root,im_path)).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, im_label


def class_from_path(path):
    _, cl, _ = path.split('/')
    return class_dict[cl]


def split_train_val_test(dataset, perc=[0.6, 0.1, 0.3]):
    train, testval = train_test_split(dataset, test_size = perc[1]+perc[2])
    val, test = train_test_split(testval, test_size = perc[2]/(perc[1]+perc[2]))
    return train, val, test

    '''





    model = resnet18(pretrained=True)
    num_class = 4
    model.fc = nn.Linear(512, num_class)
    model.num_classes = num_class
    return model


    model = resnet50(pretrained=True)
    num_class = 4
    model.fc = nn.Linear(2048, num_class)
    model.num_classes = num_class
    return model

    model_ft = models.inception_v3(pretrained=True)
    # Handle the auxilary net
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, 4)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,4)
    return model_ft

    model = squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = 4
    return model

    model = googlenet(pretrained=True)
    model.fc = nn.Linear(1024,4)
    return model
    '''

def get_model(num_class=4):
    model = resnet18(pretrained=True)
    num_class = 4
    model.fc = nn.Linear(512, num_class)
    model.num_classes = num_class
    return model


def load_checkpoint(model, optimizer=None, criterion=None, filename='checkpoint.pth.tar'):
        # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
        start_epoch = 0
        global_step = 0
        if os.path.isfile(filename):
                print("=> loading checkpoint '{}'".format(filename))
                checkpoint = torch.load(filename)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                loss_meter = checkpoint['loss_meter']
                acc_meter = checkpoint['acc_meter']
                global_step = checkpoint['global_step']
                criterion.load_state_dict(checkpoint['criterion'])
                #summary_writer = checkpoint['summary_writer']
                print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
        else:
                print("=> no checkpoint found at '{}'".format(filename))

        return model, optimizer, criterion, start_epoch, global_step, acc_meter, loss_meter

def trainval_classifier(model, train_loader, test_loader, exp_name='experiment', lr=0.01, epochs=10, momentum=0.99, logdir='logs'):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr, momentum=momentum)
    #meters
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    #writer
    writer = SummaryWriter(join(logdir, exp_name), flush_secs=1)
    print(join(logdir, exp_name))
    #device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    #definiamo un dizionario contenente i loader di training e test
    loader = {
        'train' : train_loader,
        'test' : test_loader
    }
    #inizializziamo il global step
    global_step = 0

    '''
    if load_weights_filename is not None:
        print("[*] Loading weights file: ", load_weights_filename)
        model, optimizer, criterion, e, global_step, acc_meter, loss_meter = load_checkpoint(model, optimizer, criterion, filename=load_weights_filename)
    '''
    for e in range(epochs):
        print(e)
        #iteriamo tra due modalità: train e test
        for mode in ['train','test']:

            loss_meter.reset(); acc_meter.reset()
            model.train() if mode == 'train' else model.eval()
            with torch.set_grad_enabled(mode=='train'): #abilitiamo i gradienti solo in trainin            g
                for i, batch in enumerate(loader[mode]):
                    x=batch[0].to(device) #"portiamoli sul device corretto"
                    y=batch[1].to(device)
                    output = model(x)
                    #aggiorniamo il global_step
                    #conterrà il numero di campioni visti durante il training
                    n = x.shape[0] #numero di elementi nel batch
                    global_step += n
                    l = criterion(output,y)
                    if mode=='train':
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    acc = accuracy_score(y.to('cpu'),output.to('cpu').max(1)[1])
                    loss_meter.add(l.item(),n)
                    acc_meter.add(acc,n)

                    if mode=='train':
                        writer.add_scalar('loss/train', loss_meter.value(), global_step=global_step)
                        writer.add_scalar('accuracy/train', acc_meter.value(), global_step=global_step)
                    #una volta finita l'epoca (sia nel caso di training che test, loggiamo le stime finali)
            writer.add_scalar('loss/' + mode, loss_meter.value(), global_step=global_step)
            writer.add_scalar('accuracy/' + mode, acc_meter.value(), global_step=global_step)
                    #conserviamo i pesi del modello alla fine di un ciclo di training e test
        torch.save(model.state_dict(),'./weights/%s-%d.pth'%(exp_name,e+1))

        '''
        print("[*] Saving model state dict for epoch=", e+1)
        state = {'epoch': e+1, 'global_step': global_step, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'criterion': criterion.state_dict(), 'acc_meter': acc_meter, 'loss_meter': loss_meter}
        torch.save(state, join('./weights/', '%s-%d.pth'%(exp_name,e+1)))
        e = e+1
        '''

    return model

def test_classifier(model, loader):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        predictions, labels = [], []
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            output = model(x)
            preds = output.to('cpu').max(1)[1].numpy()
            labs = y.to('cpu').numpy()
            predictions.extend(list(preds))
            labels.extend(list(labs))
        return np.array(predictions), np.array(labels)


if __name__ == '__main__':

    '''pool = Pool()
    to_factor = [ random.randint(100000, 50000000) for i in range(20)]
    results = pool.map(defs.prime_factor, to_factor)
    for value, factors in zip(to_factor, results):
        print("The factors of {} are {}".format(value, factors))'''

    train_transform = transforms.Compose([
        transforms.Resize(256), #256
        transforms.RandomCrop(224), #224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])




    #elenchiamo le classi
    classes = glob('./dataset/images/*')
    #estraiamo il nome della classe dal path completo
    classes = [basename(c) for c in classes]
    print(classes)
    class_dict = {c : i for i , c in enumerate(classes)}
    print(class_dict)
    
    image_paths = glob('dataset/*/*/*')
    print(image_paths[:10])
    print("\n")
    image_paths = ["/".join(p.split('\\')[1:]) for p in image_paths]
    print(image_paths[:10])
    labels = [class_from_path(im) for im in image_paths]
    print(labels[:10])
    dataset = pd.DataFrame({'path':image_paths, 'label':labels})
    print(dataset.head())
    
    random.seed(1395)
    np.random.seed(1359)
    train, val, test = split_train_val_test(dataset)
    
    print(len(train))
    print(len(val))
    print(len(test))
    
    train.to_csv('dataset/train.csv', index=None)
    val.to_csv('dataset/valid.csv', index=None)
    test.to_csv('dataset/test.csv', index=None)
    
    classes, ids = zip(*class_dict.items())
    classes = pd.DataFrame({'id':ids, 'class':classes}).set_index('id')
    classes.to_csv('dataset/classes.csv')
    classes = pd.read_csv('dataset/classes.csv').to_dict()['class']
    
    dataset_train = CSVImageDataset('dataset/','dataset/train.csv')
    dataset_valid = CSVImageDataset('dataset/','dataset/valid.csv')
    dataset_test = CSVImageDataset('dataset/','dataset/test.csv')
    
    im, lab = dataset_train[0]
    print('Class id:',lab, 'Class name:',classes[lab])
    print(im)
    
    dataset_train = CSVImageDataset('dataset/','dataset/train.csv', transform = train_transform)
    dataset_valid = CSVImageDataset('dataset/','dataset/valid.csv', transform = test_transform)
    dataset_test = CSVImageDataset('dataset/','dataset/test.csv', transform = test_transform)
    
    dataset_train_loader = DataLoader(dataset_train, batch_size=32, num_workers=2, shuffle=True)
    dataset_valid_loader = DataLoader(dataset_valid, batch_size=32, num_workers=2)
    dataset_test_loader = DataLoader(dataset_test, batch_size=32, num_workers=2)
    
    resnet_dataset = get_model()
    resnet_dataset_finetuned = trainval_classifier(resnet_dataset, dataset_train_loader, dataset_valid_loader, exp_name='resnet_dataset_finetuning', lr = 0.0003, epochs = 100)

    #predizioni di test del modello all'ultima epoch
    resnet_dataset_predictions_train, dataset_labels_train = test_classifier(resnet_dataset_finetuned, dataset_train_loader)
    resnet_dataset_finetuned_predictions_test, dataset_labels_test = test_classifier(resnet_dataset_finetuned, dataset_test_loader)

    print("Accuarcy di training: %0.2f%%"% (accuracy_score(dataset_labels_train, resnet_dataset_predictions_train)*100,))
    print ("Accuracy di test - ultimo modello: %0.2f%%" % (accuracy_score(dataset_labels_test, resnet_dataset_finetuned_predictions_test)*100,))
    scores_training = (precision_score(dataset_labels_train, resnet_dataset_predictions_train, average=None)*100,)
    scores_testing = (recall_score(dataset_labels_test,resnet_dataset_finetuned_predictions_test, average=None)*100,)
    print("Precision scores")
    print(scores_training)
    print("Recall scores")
    print(scores_testing)

    print("\n")
    print("Accuarcy di training: %0.2f%%"% (accuracy_score(dataset_labels_train, resnet_dataset_predictions_train)*100,))
    print ("Accuracy di test - ultimo modello: %0.2f%%" % (accuracy_score(dataset_labels_test, resnet_dataset_finetuned_predictions_test)*100,))

    print("Precision scores: %0.2f%%"% (precision_score(dataset_labels_train, resnet_dataset_predictions_train, average='micro')*100,))
    print("Recall scores: %0.2f%%"% (recall_score(dataset_labels_test,resnet_dataset_finetuned_predictions_test, average='micro')*100,))
