import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable,Function
from inception_ResNet_V2 import inception_resnet_v2

class PairwiseDistance(Function):
    '''
        compute distance of the embedding features, p is norm, when p is 2, then return L2-norm distance
    '''
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        eps = 1e-6  # in case of zeros
        diff = torch.abs(x1 - x2)     # subtraction
        out = torch.pow(diff, self.norm).sum(dim=1) # square
        return torch.pow(out + eps, 1. / self.norm) # L-p norm


class TripletLoss(Function):
    '''
       Triplet loss function.
       这里的margin就相当于是公式里的α
       loss = max(diatance(a,p) - distance(a,n) + margin, 0)
       forward method:
           args:
                anchor, positive, negative
           return:
                triplet loss
    '''
    def __init__(self, margin, num_classes=10):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.pdist = PairwiseDistance(2) # to calculate distance

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive) # distance of anchor and positive
        d_n = self.pdist.forward(anchor, negative) # distance of anchor and negative

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0) # ensure loss is no less than zero
        loss = torch.mean(dist_hinge)
        return loss


import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class TripletFaceDataset(Dataset):

    def __init__(self, root_dir, csv_name, num_triplets, transform=None):
        '''
        randomly select triplet,which means anchor,positive and negative are all selected randomly.
        args:
            root_dir : dir of data set
            csv_name : dir of train.csv
            num_triplets: total number of triplets
        '''

        self.root_dir = root_dir
        self.df = pd.read_csv(csv_name)
        self.num_triplets = num_triplets
        self.transform = transform
        self.training_triplets = self.generate_triplets(self.df, self.num_triplets)

    @staticmethod
    def generate_triplets(df, num_triplets):

        def make_dictionary_for_pic_class(df):

            '''
                make csv to the format that we want
              - pic_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
              建立类名与id一一对应的一个字典
            '''
            pic_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in pic_classes:
                    pic_classes[label] = []
                pic_classes[label].append(df.iloc[idx, 0])
            return pic_classes

        triplets = []
        classes = df['class'].unique()
        pic_classes = make_dictionary_for_pic_class(df)

        for _ in range(num_triplets):

            '''
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            '''

            pos_class = np.random.choice(classes)  # random choose positive class
            neg_class = np.random.choice(classes)  # random choose negative class
            while len(pic_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]  # get positive class's name
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]  # get negative class's name

            if len(pic_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(pic_classes[pos_class]))  # random choose anchor
                ipos = np.random.randint(0, len(pic_classes[pos_class]))  # random choose positive
                while ianc == ipos:
                    ipos = np.random.randint(0, len(pic_classes[pos_class]))
            ineg = np.random.randint(0, len(pic_classes[neg_class]))  # random choose negative

            triplets.append([pic_classes[pos_class][ianc], pic_classes[pos_class][ipos], pic_classes[neg_class][ineg],
                             pos_class, neg_class, pos_name, neg_name])

        return triplets

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        anc_img = os.path.join(self.root_dir, str(pos_name), str(anc_id) + '.png')  # join the path of anchor
        pos_img = os.path.join(self.root_dir, str(pos_name), str(pos_id) + '.png')  # join the path of positive
        neg_img = os.path.join(self.root_dir, str(neg_name), str(neg_id) + '.png')  # join the path of nagetive

        anc_img = Image.open(anc_img).convert('RGB')  # open the anchor image
        pos_img = Image.open(pos_img).convert('RGB')  # open the positive image
        neg_img = Image.open(neg_img).convert('RGB')  # open the negative image

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))  # make label transform the type we want
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))  # make label transform the type we want

        data = [anc_img, pos_img, neg_img]
        label = [pos_class, pos_class, neg_class]

        if self.transform:
            data = [self.transform(img)  # preprocessing the image
                    for img in data]

        return data, label

    def __len__(self):

        return len(self.training_triplets)


import torchvision.transforms as transforms


def train_facenet(epoch, model, optimizer, margin, num_triplets):
    model.train()
    # preprocessing function for image
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.4914, 0.4822, 0.4465]),
            std=np.array([0.2023, 0.1994, 0.2010])),
    ])

    # get dataset of triplet

    # num_triplet is adjustable
    train_set = TripletFaceDataset(root_dir='./mnist/train',
                                   csv_name='./mnist/train.csv',
                                   num_triplets=num_triplets,
                                   transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=32,
                                               shuffle=True)

    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # load data to gpu
        data[0], target[0] = data[0].cpu(), target[0].cpu()  # anchor to cuda
        data[1], target[1] = data[1].cpu(), target[1].cpu()  # positive to cuda
        data[2], target[2] = data[2].cpu(), target[2].cpu()  # negative to cuda

        data[0], target[0] = Variable(data[0]), Variable(target[0])  # anchor
        data[1], target[1] = Variable(data[1]), Variable(target[1])  # positive
        data[2], target[2] = Variable(data[2]), Variable(target[2])  # negative
        # zero setting the grad
        optimizer.zero_grad()
        # forward
        anchor = model.forward(data[0])
        positive = model.forward(data[1])
        negative = model.forward(data[2])

        # margin is adjustable
        loss = TripletLoss(margin=margin, num_classes=10).forward(anchor, positive, negative)  # get triplet loss
        total_loss += loss.item()
        # back-propagating
        loss.backward()
        optimizer.step()

    context = 'Train Epoch: {} [{}/{}], Average loss: {:.4f}'.format(
        epoch, len(train_loader.dataset), len(train_loader.dataset), total_loss / len(train_loader))
    print(context)


from sklearn import neighbors


def KNN_classifier(model, n_neighbors):
    '''
        use all train set data to make KNN classifier
    '''
    #model.eval()
    # preprocessing function for image
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])),
    ])
    # prepare dataset by ImageFolder, data should be classified by directory
    train_set = torchvision.datasets.ImageFolder(root='./mnist/train', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False)

    features, labels = [], []  # store features and labels
    for i, (data, target) in enumerate(train_loader):
        #  load data to gpu
        data, target = data.cpu(), target.cpu()
        data, target = Variable(data), Variable(target)
        # forward
        output = model(data)
        # get features and labels to make knn classifier，extend的含义是给列表追加；列表
        features.extend(output.data.cpu().numpy())
        labels.extend(target.data.cpu().numpy())

    # n_neighbor is adjustable
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(features, labels)

    return clf


def test_facenet(model, clf, test=True):
    #model.eval()
    # preprocessing function for image
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])),
    ])
    # prepare dataset by ImageFolder, data should be classified by directory
    test_set = torchvision.datasets.ImageFolder(root='./mnist/test' if test else './mnist/train', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    correct, total = 0, 0

    for i, (data, target) in enumerate(test_loader):
        # load data to gpu
        data, target = data.cpu(), target.cpu()
        data, target = Variable(data), Variable(target)
        # forward
        output = model.forward(data)
        # predict by knn classifier
        predicted = clf.predict(output.data.cpu().numpy())

        correct += (torch.tensor(predicted) == target.data.cpu()).sum()
        total += target.size(0)

    context = 'Accuracy of model in ' + ('test' if test else 'train') + \
              ' set is {}/{}({:.2f}%)'.format(correct, total, 100. * float(correct) / float(total))
    print(context)


def run_facenet():
    # hyper parameter
    lr = 0.01
    margin = 2.0
    num_triplets = 1000
    n_neighbors = 5
    embedding_size = 128
    num_epochs = 10
    # embedding_size is adjustable
    model = inception_resnet_v2(embedding_size)

    # load model into GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # define the optimizer, lr、momentum、weight_decay is adjustable
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    print('start training')
    for epoch in range(num_epochs):
        train_facenet(epoch, model, optimizer, margin, num_triplets)  # train inception_resnetV2 with triplet loss
        clf = KNN_classifier(model, n_neighbors)  # get knn classifier
        test_facenet(model, clf, False)  # validate train set
        test_facenet(model, clf, True)  # validate test set
        if (epoch + 1) % 4 == 0:
            lr = lr / 3
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
print("start")
run_facenet()