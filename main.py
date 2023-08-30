import torch 
import numpy as np
from tqdm import tqdm 
from torch.optim import Adam
import torch.nn as nn
from PIL import Image
import os

from model import CellPose
from transform import reshape_and_normalize_data, diameters, random_rotate_and_resize
from vector_gradient import labels_to_flows


def loss_fn(lbl, y, criterion, criterion2):
    """ loss function between true labels lbl and prediction y """
    veci = 5. * torch.from_numpy(lbl[:,1:]).to("cpu").float()
    lbl  = torch.from_numpy(lbl[:,0]>.5).to("cpu").float()
    loss = criterion(y[:,:2] , veci) 
    loss /= 2.
    loss2 = criterion2(y[:,2] , lbl)
    loss = loss + loss2
    return loss

def train_net(X_train, y_train, X_test, y_test, model, save_path=None, diam_mean=30, save_every=100, learning_rate=3e-10, n_epochs=200, momentum=0.9, weight_decay=1e-6, batch_size=2, rescale=True, model_name=None, device="cuda"):
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.95,0.999),weight_decay=weight_decay)
    
    criterion  = nn.MSELoss(reduction='mean')
    criterion2 = nn.BCEWithLogitsLoss(reduction='mean')

    # compute average cell diameter
    diam_train = np.array([diameters(y_train[idx]) for idx in range(len(y_train))])
    diam_train_mean = diam_train[diam_train > 0].mean()

    if rescale:
        diam_train[diam_train<5] = 5.0
        if X_test is not None:
            diam_test = np.array([(y_test[idx]) for idx in range(len(y_test))])
            diam_test[diam_test<5] = 5.0
        scale_range = 0.5
    else:
        scale_range = 1.0

    model.diam_labels.data = torch.ones(1, device=device) * diam_train_mean
    
    n_channels = X_train[0].shape[0]
    n_imgs = len(X_train)
    loss_avg, nsum = 0, 0

    if save_path is not None:
        fdir = os.path.join(save_path, "models/")

        if not os.path.exists(fdir):
            os.makedirs(fdir)
    
    model.train()
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_imgs)
        for batch in tqdm(range(0, n_imgs, batch_size)):
            inds = indices[batch:batch+batch_size]
            rsc = diam_train[inds] / diam_mean if rescale else np.ones(len(inds), np.float32)

            img, label, scale = random_rotate_and_resize(
                [X_train[idx] for idx in inds], Y=[y_train[idx][1:] for idx in inds],
                rescale=rsc, scale_range=scale_range
            )
            img = torch.from_numpy(img).float().to(device)
            optimizer.zero_grad()
            out = model(img)[0]

            loss = loss_fn(label, out, criterion, criterion2)
            loss.backward()

            train_loss = loss.item()
            optimizer.step()
            train_loss *= len(img)

            loss_avg += train_loss
            nsum += len(img)
        
        loss_avg /= nsum
        print("Epoch %d, Loss %2.4f, LR %2.4f" % (epoch, loss_avg, learning_rate))

if __name__ == "__main__":
    train_X = []
    train_y = []
    min_train_masks = 5

    data_path = "./dataset/train/"
    list_data = sorted(os.listdir(data_path))

    for fpath in tqdm (list_data):
        if "img" in fpath:
            img = np.array(Image.open(os.path.join(data_path, fpath)).convert("L"))
            mask_fpath = fpath[:3] + "_masks.png"
            mask = np.array(Image.open(os.path.join(data_path, mask_fpath)))

            train_X.append(img)
            train_y.append(mask)

    train_X, _ = reshape_and_normalize_data(train_X, channels=[0,0])
    train_flows = labels_to_flows(train_y, use_gpu=True, device="cuda")

    nmasks = np.array([label[0].max() for label in train_flows])
    nremove = (nmasks < min_train_masks).sum()
    if nremove > 0:
        ikeep = np.nonzero(nmasks >= min_train_masks)[0]
        train_X = [train_X[i] for i in ikeep]
        train_flows = [train_flows[i] for i in ikeep]

    model = CellPose(c_hiddens=[2,32,64,128,256]).to("cuda")
    train_net(train_X, train_flows, None, None, model, n_epochs=5, device="cuda")
    