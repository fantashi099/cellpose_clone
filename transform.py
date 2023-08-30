import cv2
import numpy as np


def random_rotate_and_resize(
    X,
    Y=None,
    scale_range=1.0,
    xy=(224, 224),
    do_flip=True,
    rescale=None,
    random_per_image=True,
):
    """augmentation by random rotation and resizing
    X and Y are lists or arrays of length nimg, with dims channels x Ly x Lx (channels optional)
    Parameters
    ----------
    X: LIST of ND-arrays, float
        list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]
    Y: LIST of ND-arrays, float (optional, default None)
        list of image labels of size [nlabels x Ly x Lx] or [Ly x Lx]. The 1st channel
        of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
        If Y.shape[0]==3, then the labels are assumed to be [cell probability, Y flow, X flow].
    scale_range: float (optional, default 1.0)
        Range of resizing of images for augmentation. Images are resized by
        (1-scale_range/2) + scale_range * np.random.rand()
    xy: tuple, int (optional, default (224,224))
        size of transformed images to return
    do_flip: bool (optional, default True)
        whether or not to flip images horizontally
    rescale: array, float (optional, default None)
        how much to resize images by before performing augmentations
    random_per_image: bool (optional, default True)
        different random rotate and resize per image
    Returns
    -------
    imgi: ND-array, float
        transformed images in array [nimg x nchan x xy[0] x xy[1]]
    lbl: ND-array, float
        transformed labels in array [nimg x nchan x xy[0] x xy[1]]
    scale: array, float
        amount each image was resized by
    """
    scale_range = max(0, min(2, float(scale_range)))
    nimg = len(X)
    if X[0].ndim > 2:
        nchan = X[0].shape[0]
    else:
        nchan = 1
    imgi = np.zeros((nimg, nchan, xy[0], xy[1]), np.float32)

    lbl = []
    if Y is not None:
        if Y[0].ndim > 2:
            nt = Y[0].shape[0]
        else:
            nt = 1
        lbl = np.zeros((nimg, nt, xy[0], xy[1]), np.float32)

    scale = np.ones(nimg, np.float32)

    for n in range(nimg):
        Ly, Lx = X[n].shape[-2:]

        if random_per_image or n == 0:
            # generate random augmentation parameters
            flip = np.random.rand() > 0.5
            theta = np.random.rand() * np.pi * 2
            scale[n] = (1 - scale_range / 2) + scale_range * np.random.rand()
            if rescale is not None:
                scale[n] *= 1.0 / rescale[n]
            dxy = np.maximum(
                0, np.array([Lx * scale[n] - xy[1], Ly * scale[n] - xy[0]])
            )
            dxy = (
                np.random.rand(
                    2,
                )
                - 0.5
            ) * dxy

            # create affine transform
            cc = np.array([Lx / 2, Ly / 2])
            cc1 = cc - np.array([Lx - xy[1], Ly - xy[0]]) / 2 + dxy
            pts1 = np.float32([cc, cc + np.array([1, 0]), cc + np.array([0, 1])])
            pts2 = np.float32(
                [
                    cc1,
                    cc1 + scale[n] * np.array([np.cos(theta), np.sin(theta)]),
                    cc1
                    + scale[n]
                    * np.array([np.cos(np.pi / 2 + theta), np.sin(np.pi / 2 + theta)]),
                ]
            )
            M = cv2.getAffineTransform(pts1, pts2)

        img = X[n].copy()
        if Y is not None:
            labels = Y[n].copy()
            if labels.ndim < 3:
                labels = labels[np.newaxis, :, :]

        if flip and do_flip:
            img = img[..., ::-1]
            if Y is not None:
                labels = labels[..., ::-1]
                if nt > 1:
                    labels[2] = -labels[2]

        for k in range(nchan):
            I = cv2.warpAffine(img[k], M, (xy[1], xy[0]), flags=cv2.INTER_LINEAR)
            imgi[n, k] = I

        if Y is not None:
            for k in range(nt):
                if k == 0:
                    lbl[n, k] = cv2.warpAffine(
                        labels[k], M, (xy[1], xy[0]), flags=cv2.INTER_NEAREST
                    )
                else:
                    lbl[n, k] = cv2.warpAffine(
                        labels[k], M, (xy[1], xy[0]), flags=cv2.INTER_LINEAR
                    )

            if nt > 1:
                v1 = lbl[n, 2].copy()
                v2 = lbl[n, 1].copy()
                lbl[n, 1] = -v1 * np.sin(-theta) + v2 * np.cos(-theta)
                lbl[n, 2] = v1 * np.cos(-theta) + v2 * np.sin(-theta)

    return imgi, lbl, scale


def diameters(masks):
    median = []
    for idx in range(masks.shape[0]):
        _, counts = np.unique(np.int32(masks[idx]), return_counts=True)
        counts = counts[1:]  # remove background - 0
        if len(counts) > 0:
            md = np.median(counts)
        else:
            md = 0
        md /= (np.pi**0.5) / 2
        median.append(md)
    return np.median(median)


def reshape(data, channels=[0, 0], chan_first=False):
    """reshape data using channels

    Parameters
    ----------

    data : numpy array that's (Z x ) Ly x Lx x nchan
        if data.ndim==8 and data.shape[0]<8, assumed to be nchan x Ly x Lx

    channels : list of int of length 2 (optional, default [0,0])
        First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        For instance, to train on grayscale images, input [0,0]. To train on images with cells
        in green and nuclei in blue, input [2,3].

    Returns
    -------
    data : numpy array that's (Z x ) Ly x Lx x nchan (if chan_first==False)

    """
    data = data.astype(np.float32)
    if data.ndim < 3:
        data = data[:, :, np.newaxis]
    elif data.shape[0] < 8 and data.ndim == 3:
        data = np.transpose(data, (1, 2, 0))

    # use grayscale image
    if data.shape[-1] == 1:
        data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    else:
        if channels[0] == 0:
            data = data.mean(axis=-1, keepdims=True)
            data = np.concatenate((data, np.zeros_like(data)), axis=-1)
        else:
            chanid = [channels[0] - 1]
            if channels[1] > 0:
                chanid.append(channels[1] - 1)
            data = data[..., chanid]
            for i in range(data.shape[-1]):
                if np.ptp(data[..., i]) == 0.0:
                    if i == 0:
                        print("chan to seg' has value range of ZERO")
                    else:
                        print(
                            "'chan2 (opt)' has value range of ZERO, can instead set chan2 to 0"
                        )
            if data.shape[-1] == 1:
                data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    if chan_first:
        if data.ndim == 4:
            data = np.transpose(data, (3, 0, 1, 2))
        else:
            data = np.transpose(data, (2, 0, 1))
    return data


def normalize99(Y, lower=1, upper=99):
    """normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile"""
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
    return X


def normalize_img(img, axis=-1, invert=False):
    """normalize each channel of the image so that so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities

    optional inversion

    Parameters
    ------------

    img: ND-array (at least 3 dimensions)

    axis: channel axis to loop over for normalization

    invert: invert image (useful if cells are dark instead of bright)

    Returns
    ---------------

    img: ND-array, float32
        normalized image of same size

    """
    img = img.astype(np.float32)
    img = np.moveaxis(img, axis, 0)
    for k in range(img.shape[0]):
        # ptp can still give nan's with weird images
        i99 = np.percentile(img[k], 99)
        i1 = np.percentile(img[k], 1)
        if i99 - i1 > +1e-3:  # np.ptp(img[k]) > 1e-3:
            img[k] = normalize99(img[k])
            if invert:
                img[k] = -1 * img[k] + 1
        else:
            img[k] = 0
    img = np.moveaxis(img, 0, axis)
    return img


def move_axis(img, m_axis=-1, first=True):
    """ move axis m_axis to first or last position """
    if m_axis==-1:
        m_axis = img.ndim-1
    m_axis = min(img.ndim-1, m_axis)
    axes = np.arange(0, img.ndim)
    if first:
        axes[1:m_axis+1] = axes[:m_axis]
        axes[0] = m_axis
    else:
        axes[m_axis:-1] = axes[m_axis+1:]
        axes[-1] = m_axis
    img = img.transpose(tuple(axes))
    return img


# This was edited to fix a bug where single-channel images of shape (y,x) would be 
# transposed to (x,y) if x<y, making the labels no longer correspond to the data. 
def move_min_dim(img, force=False):
    """ move minimum dimension last as channels if < 10, or force==True """
    if len(img.shape) > 2: #only makese sense to do this if channel axis is already present 
        min_dim = min(img.shape)
        if min_dim < 10 or force:
            if img.shape[-1]==min_dim:
                channel_axis = -1
            else:
                channel_axis = (img.shape).index(min_dim)
            img = move_axis(img, m_axis=channel_axis, first=False)
    return img


def reshape_and_normalize_data(
    train_data, test_data=None, channels=None, normalize=True
):
    """inputs converted to correct shapes for *training* and rescaled so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities in each channel

    Parameters
    --------------

    train_data: list of ND-arrays, float
        list of training images of size [Ly x Lx], [nchan x Ly x Lx], or [Ly x Lx x nchan]

    test_data: list of ND-arrays, float (optional, default None)
        list of testing images of size [Ly x Lx], [nchan x Ly x Lx], or [Ly x Lx x nchan]

    channels: list of int of length 2 (optional, default None)
        First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        For instance, to train on grayscale images, input [0,0]. To train on images with cells
        in green and nuclei in blue, input [2,3].

    normalize: bool (optional, True)
        normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

    Returns
    -------------

    train_data: list of ND-arrays, float
        list of training images of size [2 x Ly x Lx]

    test_data: list of ND-arrays, float (optional, default None)
        list of testing images of size [2 x Ly x Lx]

    run_test: bool
        whether or not test_data was correct size and is useable during training

    """
    # run_test = False
    for data in [train_data, test_data]:
        if data is None:
            continue
        nimg = len(data)
        for i in range(nimg):
            if channels is not None:
                # data[i] = move_min_dim(data[i], force=True)
                data[i] = reshape(data[i], channels=channels, chan_first=True)
            if data[i].ndim < 3:
                data[i] = data[i][np.newaxis, :, :]
            if normalize:
                data[i] = normalize_img(data[i], axis=0)

    # run_test = True
    return train_data, test_data


def update_axis(m_axis, to_squeeze, ndim):
    if m_axis==-1:
        m_axis = ndim-1
    if (to_squeeze==m_axis).sum() == 1:
        m_axis = None
    else:
        inds = np.ones(ndim, bool)
        inds[to_squeeze] = False
        m_axis = np.nonzero(np.arange(0, ndim)[inds]==m_axis)[0]
        if len(m_axis) > 0:
            m_axis = m_axis[0]
        else:
            m_axis = None
    return m_axis


def convert_image(data: np.array, channels, normalize=True):
    if data.ndim > 3:
        data = data.squeeze()

    if channels is not None:
        data = reshape(data, channels)
    else:
        raise ValueError("Channels error")
    
    if normalize:
        data = normalize_img(data)
    
    return data


def resize_image(img0, Ly=None, Lx=None, rsz=None, interpolation=cv2.INTER_LINEAR, no_channels=False):
    """ resize image for computing flows / unresize for computing dynamics

    Parameters
    -------------

    img0: ND-array
        image of size [Y x X x nchan] or [Lz x Y x X x nchan] or [Lz x Y x X]

    Ly: int, optional

    Lx: int, optional

    rsz: float, optional
        resize coefficient(s) for image; if Ly is None then rsz is used

    interpolation: cv2 interp method (optional, default cv2.INTER_LINEAR)

    Returns
    --------------

    imgs: ND-array 
        image of size [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

    """
    if Ly is None and rsz is None:
        error_message = 'must give size to resize to or factor to use for resizing'
        raise ValueError(error_message)

    if Ly is None:
        # determine Ly and Lx using rsz
        if not isinstance(rsz, list) and not isinstance(rsz, np.ndarray):
            rsz = [rsz, rsz]
        if no_channels:
            Ly = int(img0.shape[-2] * rsz[-2])
            Lx = int(img0.shape[-1] * rsz[-1])
        else:
            Ly = int(img0.shape[-3] * rsz[-2])
            Lx = int(img0.shape[-2] * rsz[-1])
    
    # no_channels useful for z-stacks, sot he third dimension is not treated as a channel
    # but if this is called for grayscale images, they first become [Ly,Lx,2] so ndim=3 but 
    if (img0.ndim>2 and no_channels) or (img0.ndim==4 and not no_channels):
        if Ly==0 or Lx==0:
            raise ValueError('anisotropy too high / low -- not enough pixels to resize to ratio')
        if no_channels:
            imgs = np.zeros((img0.shape[0], Ly, Lx), np.float32)
        else:
            imgs = np.zeros((img0.shape[0], Ly, Lx, img0.shape[-1]), np.float32)
        for i,img in enumerate(img0):
            imgs[i] = cv2.resize(img, (Lx, Ly), interpolation=interpolation)
    else:
        imgs = cv2.resize(img0, (Lx, Ly), interpolation=interpolation)
    return imgs


def pad_image_ND(img0, div=16, extra = 1):
    """ pad image for test-time so that its dimensions are a multiple of 16 (2D or 3D)

    Parameters
    -------------

    img0: ND-array
        image of size [nchan (x Lz) x Ly x Lx]

    div: int (optional, default 16)

    Returns
    --------------

    I: ND-array
        padded image

    ysub: array, int
        yrange of pixels in I corresponding to img0

    xsub: array, int
        xrange of pixels in I corresponding to img0

    """
    Lpad = int(div * np.ceil(img0.shape[-2]/div) - img0.shape[-2])
    xpad1 = extra*div//2 + Lpad//2
    xpad2 = extra*div//2 + Lpad - Lpad//2
    Lpad = int(div * np.ceil(img0.shape[-1]/div) - img0.shape[-1])
    ypad1 = extra*div//2 + Lpad//2
    ypad2 = extra*div//2+Lpad - Lpad//2

    if img0.ndim>3:
        pads = np.array([[0,0], [0,0], [xpad1,xpad2], [ypad1, ypad2]])
    else:
        pads = np.array([[0,0], [xpad1,xpad2], [ypad1, ypad2]])

    I = np.pad(img0,pads, mode='constant')

    Ly, Lx = img0.shape[-2:]
    ysub = np.arange(xpad1, xpad1+Ly)
    xsub = np.arange(ypad1, ypad1+Lx)
    return I, ysub, xsub