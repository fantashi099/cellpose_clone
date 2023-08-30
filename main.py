import os

import numpy as np
import torch
import torch.nn as nn
from model import CellPose
from PIL import Image
from torch.optim import Adam
from tqdm import tqdm
import cv2
from transform import (
    diameters,
    random_rotate_and_resize,
    reshape_and_normalize_data,
    convert_image,
    normalize_img,
    resize_image,
    normalize99,
    pad_image_ND,
)
from vector_gradient import labels_to_flows, to_Tensor, compute_masks
import matplotlib.pyplot as plt


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

    n_imgs = len(X_train)
    loss_avg, nsum = 0, 0

    if save_every > n_epochs:
        save_every = n_epochs - 1

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

        loss_avg /= nsum
        print("Epoch %d, Loss %2.4f, LR %2.5f" % (epoch, loss_avg, learning_rate))
        if epoch % eval_step == 0:
            if X_test is not None and y_test is not None:
                loss_avg_test, nsum = 0, 0
                eval_imgs = len(X_test)
                indices = np.random.permutation(eval_imgs)
                for batch in range(0, eval_imgs, eval_batch_size):
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

                loss_avg_test /= nsum
                print("Eval Loss %2.4f" % (loss_avg_test))

        if save_path is not None:
            if epoch % save_every == 0 and epoch > 0:
                if model_name is None:
                    model_name = "default"
                file_name = "model_{}_epoch_{}".format(model_name, epoch)
                fpath = os.path.join(fdir, file_name)
                print(f"Save model in epoch: {epoch}")
                model.save_model(fpath)


# modified to use sinebow color
def dx_to_circ(dP,transparency=False,mask=None):
    """ dP is 2 x Y x X => 'optic' flow representation 
    
    Parameters
    -------------
    
    dP: 2xLyxLx array
        Flow field components [dy,dx]
        
    transparency: bool, default False
        magnitude of flow controls opacity, not lightness (clear background)
        
    mask: 2D array 
        Multiplies each RGB component to suppress noise
    
    """
    
    dP = np.array(dP)
    mag = np.clip(normalize99(np.sqrt(np.sum(dP**2,axis=0))), 0, 1.)
    angles = np.arctan2(dP[1], dP[0])+np.pi
    a = 2
    r = ((np.cos(angles)+1)/a)
    g = ((np.cos(angles+2*np.pi/3)+1)/a)
    b =((np.cos(angles+4*np.pi/3)+1)/a)
    
    if transparency:
        im = np.stack((r,g,b,mag),axis=-1)
    else:
        im = np.stack((r*mag,g*mag,b*mag),axis=-1)
        
    if mask is not None and transparency and dP.shape[0]<3:
        im[:,:,-1] *= mask
        
    im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    return im


def preprocess(
    data,
    model,
    do_masks=True,
    normalize=True,
    rescale=1.0,
    augment=False,
    tile=True,
    tile_overlap=0.1,
    flow_threshold=0.4,
    min_size=15,
    cellprob_threshold=0.0,
    resample=True,
    interp=True,
    use_gpu=True,
    device="cuda",
):
    shape = data.shape
    nimg = shape[0]

    styles = np.zeros((nimg, 256), np.float32)

    if resample:
        dP = np.zeros((2, nimg, shape[1], shape[2]), np.float32)
        cellprob = np.zeros((nimg, shape[1], shape[2]), np.float32)
    else:
        dP = np.zeros(
            (2, nimg, int(shape[1] * rescale), int(shape[2] * rescale)), np.float32
        )
        cellprob = np.zeros(
            (nimg, int(shape[1] * rescale), int(shape[2] * rescale)), np.float32
        )

    for idx in range(nimg):
        img = data[idx]
        if normalize:
            img = normalize_img(img)
        if rescale != 1.0:
            img = resize_image(img, rsz=rescale)

        # make image nchan x Ly x Lx for net
        img = np.transpose(img, (2,0,1))
        detranspose = (1,2,0)

        # pad image for net so w and h are divisible by 4
        img, ysub, xsub = pad_image_ND(img)
        # slices from padding
        slc = [slice(0, img.shape[n]+1) for n in range(img.ndim)]
        slc[-3] = slice(0, 3 + 32*False + 1)
        slc[-2] = slice(ysub[0], ysub[-1]+1)
        slc[-1] = slice(xsub[0], xsub[-1]+1)
        slc = tuple(slc)

        # Model Inference
        img = np.expand_dims(img, axis=0)
        img = to_Tensor(img, device)
        model.eval()
        with torch.no_grad():
            yf, style = model(img)
            yf = yf.detach().cpu().numpy()
            style = style.detach().cpu().numpy()
        yf, style = yf[0], style[0]

        style /= (style**2).sum()**0.5

        yf = yf[slc]
        yf = np.transpose(yf, detranspose)

        # plt.figure(figsize=(10,10))
        # ax = plt.subplot(2,2,1)
        # plt.imshow(yf[..., 0])
        # plt.savefig("abc.png")
        # ax = plt.subplot(2,2,2)
        # plt.imshow(yf[..., 1])
        # ax = plt.subplot(2,2,3)
        # plt.imshow(yf[..., 2])
        # plt.show()
        # print(yf[..., 0][yf[..., 0]>0])
        # print(yf[..., 1][yf[..., 1]>0])
        # print(yf[..., 2][yf[..., 2]>0])


        if resample:
            yf = resize_image(yf, shape[1], shape[2])

        cellprob[idx] = yf[:, :, 0]
        dP[:, idx] = yf[:, :, 1:].transpose((2, 0, 1))
        styles[idx] = style
    styles = styles.squeeze()

    niter = 200 / rescale
    masks, p = [], []
    resize = [shape[1], shape[2]] if not resample else None
    for idx in range(nimg):
        outputs = compute_masks(
            dP[:, idx],
            cellprob[idx],
            niter=niter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            interp=interp,
            resize=resize,
            use_gpu=use_gpu,
            device=device,
        )
        masks.append(outputs[0])
        p.append(outputs[1])

    masks = np.array(masks)
    p = np.array(p)

    masks, dP, cellprob, p = (
        masks.squeeze(),
        dP.squeeze(),
        cellprob.squeeze(),
        p.squeeze(),
    )
    return masks, styles, dP, cellprob, p


def predict(
    data,
    model,
    batch_size=8,
    channels=None,
    diam_mean=30,
    diameter=30,
    augment=False,
    tile=True,
    tile_overlap=0.1,
    resample=True,
    interp=True,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    do_masks=True,
    min_size=15,
    stitch_threshold=0.0,
    use_gpu=False,
    device="cpu"
):
    data = convert_image(data, channels=channels, normalize=False)

    if data.ndim < 4:
        data = data[np.newaxis, ...]

    rescale = diam_mean / diameter

    masks, styles, dP, cellprob, p = preprocess(
        data,
        model,
        do_masks=do_masks,
        normalize=True,
        rescale=rescale,
        augment=augment,
        tile=tile,
        tile_overlap=tile_overlap,
        flow_threshold=flow_threshold,
        min_size=min_size,
        cellprob_threshold=cellprob_threshold,
        resample=resample,
        interp=interp,
        use_gpu=use_gpu,
        device=device,
    )
    flows = [dx_to_circ(dP), dP, cellprob, p]
    return masks, flows, styles


def overlay(image, mask, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = list(np.random.choice(range(256), size=3))
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


if __name__ == "__main__":
    # train_X, train_y = [], []
    # test_X, test_y = [], []
    # min_train_masks = 5

    # data_path = "./dataset/train/"
    # list_data = sorted(os.listdir(data_path))

    # print("Load train data")
    # for fpath in tqdm(list_data):
    #     if "img" in fpath:
    #         img = np.array(Image.open(os.path.join(data_path, fpath)).convert("L"))
    #         mask_fpath = fpath[:3] + "_masks.png"
    #         mask = np.array(Image.open(os.path.join(data_path, mask_fpath)))

    #         train_X.append(img)
    #         train_y.append(mask)

    # data_path = "./dataset/test/"
    # list_data = sorted(os.listdir(data_path))

    # print("Load test data")
    # for fpath in tqdm(list_data):
    #     if "img" in fpath:
    #         img = np.array(Image.open(os.path.join(data_path, fpath)).convert("L"))
    #         mask_fpath = fpath[:3] + "_masks.png"
    #         mask = np.array(Image.open(os.path.join(data_path, mask_fpath)))

    #         test_X.append(img)
    #         test_y.append(mask)

    # train_X, test_X = reshape_and_normalize_data(
    #     train_X, test_data=test_X, channels=[0, 0]
    # )
    # print("Create Vector Gradient from Label Masks")
    # train_flows = labels_to_flows(train_y, use_gpu=True, device="cpu")
    # test_flows = labels_to_flows(test_y, use_gpu=True, device="cpu")

    # nmasks = np.array([label[0].max() for label in train_flows])
    # nremove = (nmasks < min_train_masks).sum()
    # if nremove > 0:
    #     ikeep = np.nonzero(nmasks >= min_train_masks)[0]
    #     train_X = [train_X[i] for i in ikeep]
    #     train_flows = [train_flows[i] for i in ikeep]

    model = CellPose(c_hiddens=[2, 32, 64, 128, 256]).to("cuda")
    # print("Start Training")
    # train_net(
    #     train_X,
    #     train_flows,
    #     test_X,
    #     test_flows,
    #     model,
    #     n_epochs=2,
    #     save_path="./",
    #     device="cuda",
    # )

    img = Image.open("./dataset/test/001_img.png")
    gray_img = np.array(img.convert("L"))

    print("Start Predicting")
    model.load_model("./models/model_default_epoch_100", device="cuda")
    masks, flows, styles = predict(gray_img, model, channels=[0,0], use_gpu=True, device="cuda")

    # print(masks[masks>0])
    img_overlay = overlay(np.array(img), masks, alpha=0.5)
    
    plt.figure(figsize=(10,10))
    plt.imshow(img_overlay)
    plt.show()