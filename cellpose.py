import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from model import CellPose
from transform import (average_tiles, convert_image, diameters, make_tiles,
                       normalize_img, pad_image_ND, random_rotate_and_resize,
                       reshape_and_normalize_data, resize_image)
from vector_gradient import (compute_masks, dx_to_circ, labels_to_flows,
                             to_Tensor)


class CellPoseModel:
    def __init__(
        self,
        c_hiddens: list = [2, 32, 64, 128, 256],
        diam_mean: float = 30,
        pretrained_model: str = "",
        device: str = "cpu",
    ) -> None:
        self.cellpose = CellPose(c_hiddens=c_hiddens, diam_mean=diam_mean).to(device)

        self.diam_mean = diam_mean
        self.diam_labels = diam_mean
        self.device = device

        if os.path.exists(pretrained_model):
            print(f"Load pre-trained model at: {pretrained_model}")
            self.cellpose.load_model(pretrained_model, device=self.device)

            self.diam_mean = self.cellpose.diam_mean.data.cpu().numpy()[0]
            self.diam_labels = self.cellpose.diam_labels.data.cpu().numpy()[0]
            
            sample = torch.rand(2, 64, 64).unsqueeze(0).to(self.device)
            _ = self.cellpose(sample)
            del _
        else:
            print("Not provide valid pre-trained path, load model from scracth")

    def set_device(self, device: str = "cpu"):
        self.cellpose.to(device)

    def _get_batch_imgs(
        self, data, label, batch, batch_size, indices, diam, rescale, scale_range
    ):
        inds = indices[batch : batch + batch_size]
        rsc = diam[inds] / self.diam_mean if rescale else np.ones(len(inds), np.float32)

        img, label, scale = random_rotate_and_resize(
            [data[idx] for idx in inds],
            Y=[label[idx][1:] for idx in inds],
            rescale=rsc,
            scale_range=scale_range,
        )
        img = to_Tensor(img, self.device).float()
        return img, label

    def loss_fn(self, lbl, y):
        """loss function between true labels lbl and prediction y"""
        veci = 5.0 * to_Tensor(lbl[:, 1:], self.device).float()
        lbl = to_Tensor(lbl[:, 0] > 0.5, self.device).float()
        loss = self.criterion(y[:, :2], veci)
        loss /= 2.0
        loss2 = self.criterion2(y[:, 2], lbl)
        loss = loss + loss2
        return loss

    def _net_infer(self, img):
        # Model Inference
        img = to_Tensor(img, self.device)
        self.cellpose.eval()
        with torch.no_grad():
            yf, style = self.cellpose(img)
            yf = yf.detach().cpu().numpy()
            style = style.detach().cpu().numpy()
        return yf, style

    def train(
        self,
        X_train: list,
        y_train: list,
        X_test: list = None,
        y_test: list = None,
        channels: list = [0, 0],
        use_gpu: bool = True,
        save_path: str = None,
        min_train_masks: int = 5,
        save_every: int = 50,
        eval_step: int = 50,
        learning_rate: float = 3e-4,
        n_epochs: int = 200,
        weight_decay: float = 1e-6,
        batch_size: int = 2,
        eval_batch_size: int = 1,
        rescale: bool = True,
        model_name: str = None,
    ) -> None:
        if len(X_train) != len(y_train):
            raise ValueError("train data and labels are not same length!")

        if X_test is not None and y_test is not None:
            if (len(X_test) != len(y_test)):
                raise ValueError("test data and labels are not same length!")
        
        if channels is None:
            print("Channels is None => Set channels = [0,0]")
            channels = [0,0]

        X_train, X_test = reshape_and_normalize_data(
            X_train, test_data=X_test, channels=channels
        )

        print("Create Vector Gradient from Label Masks")
        train_flows = labels_to_flows(y_train, use_gpu=use_gpu, device=self.device)
        if y_test is not None:
            test_flows = labels_to_flows(y_test, use_gpu=use_gpu, device=self.device)

        nmasks = np.array([label[0].max() for label in train_flows])
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            X_train = [X_train[i] for i in ikeep]
            train_flows = [train_flows[i] for i in ikeep]

        self.optimizer = Adam(
            self.cellpose.parameters(),
            lr=learning_rate,
            betas=(0.95, 0.999),
            weight_decay=weight_decay,
        )

        self.criterion = nn.MSELoss(reduction="mean")
        self.criterion2 = nn.BCEWithLogitsLoss(reduction="mean")

        # compute average cell diameter
        diam_train = np.array(
            [diameters(train_flows[idx][0])[0] for idx in range(len(train_flows))]
        )
        diam_train_mean = diam_train[diam_train > 0].mean()
        self.diam_labels = diam_train_mean

        if rescale:
            diam_train[diam_train < 5] = 5.0
            if X_test is not None and y_test is not None:
                diam_test = np.array(
                    [diameters(test_flows[idx][0])[0] for idx in range(len(test_flows))]
                )
                diam_test[diam_test < 5] = 5.0
            scale_range = 0.5
        else:
            scale_range = 1.0

        self.cellpose.diam_labels.data = (
            torch.ones(1, device=self.device) * diam_train_mean
        )

        n_imgs = len(X_train)
        loss_avg, nsum = 0, 0

        if save_every > n_epochs:
            save_every = n_epochs

        if save_path is not None:
            fdir = os.path.join(save_path, "models/")

            if not os.path.exists(fdir):
                os.makedirs(fdir)

        print("Start Training Model")
        for epoch in range(n_epochs+1):
            indices = np.random.permutation(n_imgs)
            for batch in tqdm(range(0, n_imgs, batch_size)):
                img, label = self._get_batch_imgs(
                    X_train,
                    train_flows,
                    batch,
                    batch_size,
                    indices,
                    diam_train,
                    rescale,
                    scale_range,
                )

                self.optimizer.zero_grad()
                self.cellpose.train()

                out = self.cellpose(img)[0]
                loss = self.loss_fn(label, out)
                loss.backward()
                self.optimizer.step()

                train_loss = loss.item()
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
                        img, label = self._get_batch_imgs(
                            X_test,
                            test_flows,
                            batch,
                            eval_batch_size,
                            indices,
                            diam_test,
                            rescale,
                            scale_range,
                        )

                        self.cellpose.eval()
                        with torch.no_grad():
                            out = self.cellpose(img)[0]
                            loss = self.loss_fn(label, out)
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
                    file_name = "model_{}_epoch_{}.pt".format(model_name, epoch)
                    fpath = os.path.join(fdir, file_name)
                    print(f"Save model in epoch: {epoch} - Saved path: {fpath}")
                    self.cellpose.save_model(fpath)

    def postprocess(
        self,
        data,
        normalize: bool = True,
        rescale: float = 1.0,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        resample: bool = True,
        interp: bool = True,
        use_gpu: bool = True,
        invert: bool = False,
        tile: bool = True,
        batch_infer: int = 1,
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
            if normalize or invert:
                img = normalize_img(img, invert=invert)
            if rescale != 1.0:
                img = resize_image(img, rsz=rescale)

            # Model Inference
            # make image nchan x Ly x Lx for net
            img = np.transpose(img, (2, 0, 1))
            detranspose = (1, 2, 0)

            # pad image for net so w and h are divisible by 4
            img, ysub, xsub = pad_image_ND(img)
            # slices from padding
            slc = [slice(0, img.shape[n] + 1) for n in range(img.ndim)]
            slc[-3] = slice(0, 3 + 32 * False + 1)
            slc[-2] = slice(ysub[0], ysub[-1] + 1)
            slc[-1] = slice(xsub[0], xsub[-1] + 1)
            slc = tuple(slc)

            # =========================================

            if tile:
                IMG, ysub, xsub, Ly, Lx = make_tiles(
                    img, bsize=224, tile_overlap=0.1
                )
                ny, nx, nchan, ly, lx = IMG.shape
                IMG = np.reshape(IMG, (ny * nx, nchan, ly, lx))
                niter = int(np.ceil(IMG.shape[0] / batch_infer))
                nout = 3 + 32 * False
                y = np.zeros((IMG.shape[0], nout, ly, lx))
                for k in range(niter):
                    irange = slice(
                        batch_infer * k, min(IMG.shape[0], batch_infer * k + batch_infer)
                    )
                    yf, tiled_style = self._net_infer(IMG[irange])
                    y[irange] = yf.reshape(
                        irange.stop - irange.start, yf.shape[-3], yf.shape[-2], yf.shape[-1]
                    )
                    if k == 0:
                        style = tiled_style[0]
                    style += tiled_style.sum(axis=0)
                style /= IMG.shape[0]

                yf = average_tiles(y, ysub, xsub, Ly, Lx)
                yf = yf[:, : img.shape[1], : img.shape[2]]
            else:
                # Inference the whole image
                img = np.expand_dims(img, axis=0)
                yf, style = self._net_infer(img)
                yf, style = yf[0], style[0]

            # =====================================================================
            style /= (style**2).sum() ** 0.5

            yf = yf[slc]
            yf = np.transpose(yf, detranspose)

            if resample:
                yf = resize_image(yf, shape[1], shape[2])

            cellprob[idx] = yf[:, :, 2]
            dP[:, idx] = yf[:, :, :2].transpose((2, 0, 1))
            styles[idx] = style
        styles = styles.squeeze()

        # Compute Mask from Prediction
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
                device=self.device,
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
        self,
        data,
        channels=None,
        diameter=30,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        resample=True,
        interp=True,
        use_gpu=False,
        invert: bool = False,
        tile: bool = True,
        batch_infer: int = 1,
    ):
        if channels is None:
            print("Channels is None => Set channels = [0,0]")
            channels = [0,0]

        # reshape image (normalization happens in postprocess)
        data = convert_image(data, channels=channels, normalize=False)

        if data.ndim < 4:
            data = data[np.newaxis, ...]

        if diameter is not None and diameter > 0:
            rescale = self.diam_mean / diameter
        else:
            rescale = self.diam_mean / self.diam_labels

        start_time = time.time()
        masks, styles, dP, cellprob, p = self.postprocess(
            data,
            normalize=True,
            rescale=rescale,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            resample=resample,
            invert=invert,
            interp=interp,
            use_gpu=use_gpu,
            tile=tile,
            batch_infer=batch_infer,
        )
        flows = [dx_to_circ(dP), dP, cellprob, p]
        end_time = time.time()
        print(
            f"Total Process Inference Time: {end_time - start_time}, FPS: {1/(end_time - start_time)}"
        )
        return masks, flows, styles

    def save_model(self, fpath: str):
        if not fpath.endswith(".pt"):
            fpath += ".pt"
        print(f"Saved path: {fpath}")
        self.cellpose.save_model(fpath)
