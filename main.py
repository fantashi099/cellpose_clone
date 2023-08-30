import os

import numpy as np
import torch
import torch.nn as nn
from model import CellPose
from PIL import Image
from torch.optim import Adam
from tqdm import tqdm
from transform import (diameters, random_rotate_and_resize,
                       reshape_and_normalize_data)
from vector_gradient import labels_to_flows, to_Tensor


def loss_fn(lbl, y, criterion, criterion2, device):
    """loss function between true labels lbl and prediction y"""
    veci = 5.0 * to_Tensor(lbl[:, 1:], device).float()
    lbl = to_Tensor(lbl[:, 0] > 0.5, device).float()
    loss = criterion(y[:, :2], veci)
    loss /= 2.0
    loss2 = criterion2(y[:, 2], lbl)
    loss = loss + loss2
    return loss


def train_net(
    X_train,
    y_train,
    X_test,
    y_test,
    model,
    save_path=None,
    diam_mean=30,
    save_every=100,
    eval_step=50,
    learning_rate=3e-4,
    n_epochs=200,
    weight_decay=1e-6,
    batch_size=2,
    eval_batch_size=1,
    rescale=True,
    model_name=None,
    device="cuda",
):
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.95, 0.999),
        weight_decay=weight_decay,
    )

    criterion = nn.MSELoss(reduction="mean")
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")

    # compute average cell diameter
    diam_train = np.array([diameters(y_train[idx]) for idx in range(len(y_train))])
    diam_train_mean = diam_train[diam_train > 0].mean()

    if rescale:
        diam_train[diam_train < 5] = 5.0
        if X_test is not None:
            diam_test = np.array([diameters(y_test[idx]) for idx in range(len(y_test))])
            diam_test[diam_test < 5] = 5.0
        scale_range = 0.5
    else:
        scale_range = 1.0

    model.diam_labels.data = torch.ones(1, device=device) * diam_train_mean

    n_channels = X_train[0].shape[0]
    n_imgs = len(X_train)
    loss_avg, nsum = 0, 0

    if save_every > n_epochs:
        save_every = n_epochs

    if save_path is not None:
        fdir = os.path.join(save_path, "models/")

        if not os.path.exists(fdir):
            os.makedirs(fdir)

    for epoch in range(n_epochs):
        model.train()
        indices = np.random.permutation(n_imgs)
        for batch in tqdm(range(0, n_imgs, batch_size)):
            inds = indices[batch : batch + batch_size]
            rsc = (
                diam_train[inds] / diam_mean
                if rescale
                else np.ones(len(inds), np.float32)
            )

            img, label, scale = random_rotate_and_resize(
                [X_train[idx] for idx in inds],
                Y=[y_train[idx][1:] for idx in inds],
                rescale=rsc,
                scale_range=scale_range,
            )
            img = to_Tensor(img, device).float()
            optimizer.zero_grad()
            out = model(img)[0]

            loss = loss_fn(label, out, criterion, criterion2, device)
            loss.backward()

            train_loss = loss.item()
            optimizer.step()
            train_loss *= len(img)

            loss_avg += train_loss
            nsum += len(img)

        print("Epoch %d, Loss %2.4f, LR %2.5f" % (epoch, loss_avg, learning_rate))
        if epoch % eval_step == 0:
            loss_avg /= nsum
            if X_test is not None and y_test is not None:
                loss_avg_test, nsum = 0, 0
                n_imgs = len(X_test)
                indices = np.random.permutation(n_imgs)
                for batch in range(0, n_imgs, eval_batch_size):
                    inds = indices[batch : batch + eval_batch_size]
                    rsc = (
                        diam_test[inds] / diam_mean
                        if rescale
                        else np.ones(len(inds), np.float32)
                    )
                    img, label, scale = random_rotate_and_resize(
                        [X_test[idx] for idx in inds],
                        Y=[y_test[idx][1:] for idx in inds],
                        rescale=rsc,
                        scale_range=scale_range,
                    )
                    img = to_Tensor(img, device).float()
                    model.eval()
                    with torch.no_grad():
                        out = model(img)[0]
                        loss = loss_fn(label, out, criterion, criterion2, device)
                        test_loss = loss.item()
                        test_loss *= len(img)
                        
                        loss_avg_test += test_loss
                        nsum += len(img)
                print("Eval Loss %2.4f, LR %2.5f" % (loss_avg_test, learning_rate))

        if save_path is not None:
            if epoch == save_every:
                if model_name is None:
                    model_name = "default"
                file_name = "model_{}_epoch_{}".format(model_name, epoch)
                fpath = os.path.join(fdir, file_name)
                print(f"Save model in epoch: {epoch}")
                model.save_model(fpath)


if __name__ == "__main__":
    train_X, train_y = [], []
    test_X, test_y = [], []
    min_train_masks = 5

    data_path = "./dataset/train/"
    list_data = sorted(os.listdir(data_path))

    print("Load train data")
    for fpath in tqdm(list_data):
        if "img" in fpath:
            img = np.array(Image.open(os.path.join(data_path, fpath)).convert("L"))
            mask_fpath = fpath[:3] + "_masks.png"
            mask = np.array(Image.open(os.path.join(data_path, mask_fpath)))

            train_X.append(img)
            train_y.append(mask)

    data_path = "./dataset/test/"
    list_data = sorted(os.listdir(data_path))

    print("Load test data")
    for fpath in tqdm(list_data):
        if "img" in fpath:
            img = np.array(Image.open(os.path.join(data_path, fpath)).convert("L"))
            mask_fpath = fpath[:3] + "_masks.png"
            mask = np.array(Image.open(os.path.join(data_path, mask_fpath)))

            test_X.append(img)
            test_y.append(mask)


    train_X, test_X = reshape_and_normalize_data(train_X, test_data=test_X, channels=[0, 0])
    print("Create Vector Gradient from Label Masks")
    train_flows = labels_to_flows(train_y, use_gpu=True, device="cuda")
    test_flows = labels_to_flows(test_y, use_gpu=True, device="cuda")

    nmasks = np.array([label[0].max() for label in train_flows])
    nremove = (nmasks < min_train_masks).sum()
    if nremove > 0:
        ikeep = np.nonzero(nmasks >= min_train_masks)[0]
        train_X = [train_X[i] for i in ikeep]
        train_flows = [train_flows[i] for i in ikeep]

    

    model = CellPose(c_hiddens=[2, 32, 64, 128, 256]).to("cuda")
    print("Start Training")
    train_net(train_X, train_flows, test_X, test_flows, model, n_epochs=2, save_path="./", device="cuda")
