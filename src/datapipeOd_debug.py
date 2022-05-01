import os
import json
from matplotlib.pyplot import imshow
import numpy as np
import tensorflow as tf



def readJsonAnnotation(jsonfile, datapath, classNames, minLength=10):
    """Read JSON Annotation (PASCAL VOC)"""
    with open(jsonfile, 'r') as f1:
        data = json.load(f1)

    imgpath = os.path.join(datapath, data["imagePath"])

    size = [data["imageWidth"], data["imageHeight"]]

    boxes, labels = [], []

    for obj in data['shapes']:
        if obj["shape_type"] == "rectangle":
            label = obj['label']

            if label not in classNames:
                continue

            x1 = min((obj['points'][0][0], obj['points'][1][0])) / data["imageWidth"]
            x2 = max((obj['points'][0][0], obj['points'][1][0])) / data["imageWidth"]
            y1 = min((obj['points'][0][1], obj['points'][1][1])) / data["imageHeight"]
            y2 = max((obj['points'][0][1], obj['points'][1][1])) / data["imageHeight"]

            # Dont consider when too small
            if np.sqrt(size[0]*size[1]*(x2-x1)*(y2-y1)) < minLength:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(classNames.index(label))

    if len(boxes) == 0:
        boxes = np.zeros((0, 4))

    return imgpath, boxes, labels, size




class Datapipe:
    def __init__(self, datapath, classNames=[]):


        # Path where annotation jsons reside
        self.datapath = datapath
        self.classNames = classNames

        # Find all json files in datapath
        self.filenames = self.getFileNames(datapath)

  
    # ============================
    def getFileNames(self, datapath):
        filenames = []
        for root, dirs, files in os.walk(datapath):
            for file in files:
                if file.endswith(".json"):
                     filenames.append(os.path.join(root, file))
        return filenames


    # ============================

    @property
    def nd(self):
        return len(self.filenames)

    @property
    def nc(self):
        return len(self.classNames)

    @property
    def cdict(self):
        return {k:v for v,k in enumerate(self.classNames)}


    # ============================
    def create(self, nx, ny, iw, ih, ic, batchSize, sigma=0.02,
               shuffle_buffer_size=5000, nrepeat=1, minBoxSize=6):

        """Creates the datapipe"""

        self.nx = nx
        self.ny = ny
        self.iw = iw
        self.ih = ih
        self.ic = ic
        self.minBoxSize = minBoxSize
        self.sigma = sigma 

        # Let's build the pipeline
        dataset = tf.data.Dataset.from_tensor_slices(self.filenames)

        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat(nrepeat)

        # Load the Json
        dataset = dataset.map(self._loadJson)

        # Load the image
        # dataset = dataset.map(self._processLoadImagePatchWise)
        dataset = dataset.map(self._processLoadImage)

        dataset = dataset.map(self._gaussianLabel)
        # Augment the image
        # if True:
        #     dataset = dataset.map(self._processAddNoise)
        # if True:
        #     dataset = dataset.map(self._processAugmentFlip)
        # if True:
        #     dataset = dataset.map(self._processAugmentFlipVertically)
        # if True:
        #     dataset = dataset.map(self._processAugmentColor)
        # if True:
        #     dataset = dataset.map(self._processAugmentCrop)
        # if True:
        #     dataset = dataset.map(self._processRotate)





        # Finally calculate ground truth. Must be last step
    #    dataset = dataset.map(self._processCalculateGroundTruth)

        # Apply batching
        dataset = dataset.batch(batchSize)

        # Prefetching
      #  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


    # ============================
    def _loadJson(self, jsonfile):

        def _pyLoadJson(content):
            imgpath, boxes, labels, size = readJsonAnnotation(
                jsonfile=content.numpy().decode("utf-8"),
                datapath=self.datapath,
                classNames=self.classNames,
                minLength=self.minBoxSize
            )
            return imgpath, boxes, labels, size

        imgpath, boxes, labels, size = tf.py_function(
            _pyLoadJson, [jsonfile], [tf.string, tf.float32, tf.int32, tf.int32]
        )

        return imgpath, boxes, labels, jsonfile

    # ============================
    def _processLoadImage(self, imgpath, boxes, labels, jsonfile):

        ih, iw = self.ih, self.iw
        ic = self.ic

        img = tf.io.read_file(imgpath)
        img = tf.image.decode_jpeg(img, channels=ic)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (ih, iw))

        return img, boxes, labels, jsonfile

    # ============================
    def _processLoadImagePatchWise(self, imgpath, boxes, labels, jsonfile):

        px = self.px
        py = self.py

        ih, iw = self.ih, self.iw
        ic = self.ic

        # Load original image
        img = tf.io.read_file(imgpath)
        img = tf.image.decode_jpeg(img, channels=ic)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (py * ih, px * iw))

        # Calculate box dimensions
        bwh = tf.transpose(tf.stack((
          (boxes[:, 2] - boxes[:, 0]),
          (boxes[:, 3] - boxes[:, 1])
        )))
        # Calculate box centroids and best matching patch
        bxyc = tf.transpose(tf.stack((
            0.5 * (boxes[:, 0] + boxes[:, 2]),
            0.5 * (boxes[:, 1] + boxes[:, 3])
        )))

        pxcy = getGrid(px, py)
        # Difference boxes to centeroids
        dist = tf.reduce_sum(tf.square((tf.expand_dims(bxyc, 1) - pxcy)), -1)

        # Grid i,j indices of best matching cell
        ixyc = tf.math.argmin(dist, axis=-1, output_type=tf.dtypes.int32)

        # Create patches of image, stride by half corruption size
        img = tf.image.extract_patches(images=tf.expand_dims(img, 0),
                                       sizes=[1, ih, iw, 1],
                                       strides=[1, ih, iw, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='VALID')

        img = tf.reshape(img, (-1, ih, iw, ic))

        # Draw one random patch for training. For inference a batch of patches is created
        idx = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(img)[0], dtype=tf.int32)

        # Calculate rescaled bounding boxes
        bxyc_offset = tf.expand_dims(pxcy[idx,:] - tf.constant([0.5/px,0.5/py]),0)
        scale = tf.expand_dims(tf.constant([px, py], dtype=tf.float32), 0)
        ibrel = tf.where(tf.math.equal(ixyc, idx))
        bxyc = scale*(tf.gather_nd(bxyc, ibrel) - bxyc_offset)
        bwh = scale*tf.gather_nd(bwh,ibrel)

        boxes = tf.concat((
            tf.expand_dims(bxyc[:, 0] - 0.5 * bwh[:, 0], -1),
            tf.expand_dims(bxyc[:, 1] - 0.5 * bwh[:, 1], -1),
            tf.expand_dims(bxyc[:, 0] + 0.5 * bwh[:, 0], -1),
            tf.expand_dims(bxyc[:, 1] + 0.5 * bwh[:, 1], -1),
        ), axis = -1)

        # Adjust labels
        labels = tf.gather_nd(labels, ibrel)

        return img[idx, :, :, :], boxes, labels, jsonfile


    # ============================
    def _processRotate(self, img, boxes, labels, jsonfile):


        if self.iw == self.ih:
            def rotate(img, boxes):
                img = tf.transpose(img,[1,0,2])
                boxes = tf.stack([boxes[:,1], boxes[:,0], boxes[:,3], boxes[:,2]], 1)
               # boxes = tf.transpose(boxes, [1,0,3,2])
                return img, boxes

            def norotate(img, boxes):
                return img, boxes

            choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
            img, boxes = tf.cond(choice < 0.5,
                lambda: rotate(img, boxes),
                lambda: norotate(img, boxes)
            )


        return img, boxes, labels, jsonfile


    # ============================
    def _processAugmentColor(self, img, boxes, labels, jsonfile, rand_saturation=[0,1], rand_contrast=[0.6,1.1], rand_hue=0.5, rand_brightness=0.5, **kwargs):

        if self.ic == 3:
            img = tf.image.random_hue(img, rand_hue)
            img = tf.image.random_saturation(
                img, 
                rand_saturation[0], 
                rand_saturation[1]
            )

        img = tf.image.random_brightness(img, )
        img = tf.image.random_contrast(img, 
            rand_contrast[0],
            rand_contrast[1]
        )
        return img, boxes, labels, jsonfile


   # ============================
    def _processAugmentCrop(self, img, boxes, labels, jsonfile, **kwargs):

        # Generate 20 crop settings, ranging from a 1% to 20% crop.
        def cropimage(img, boxes, width, height):
            scales = np.arange(
                self.optionsDict["augs"]["crop"]["rand_scales"][0],
                self.optionsDict["augs"]["crop"]["rand_scales"][1],
                self.optionsDict["augs"]["crop"]["rand_scales"][2]
            )
            cropboxes = np.zeros((len(scales), 4))
            for i, scale in enumerate(scales):
                cx1 = cy1 = 0.5 - (0.5 * scale)
                cx2 = cy2 = 0.5 + (0.5 * scale)
                cropboxes[i] = [cx1, cy1, cx2, cy2]

            cropboxes = tf.convert_to_tensor(cropboxes, dtype=tf.float32)

            # Create different crops for an image
            crops = tf.image.crop_and_resize(
                [img],
                boxes=cropboxes,
                box_indices=np.zeros(cropboxes.shape[0]),
                crop_size=(height, width)
            )

            # Return a random crop
            idx = tf.random.uniform(shape=[], minval=0, maxval=cropboxes.shape[0], dtype=tf.int32)

            # Resize bbox
            scale = tf.stack([
                (cropboxes[idx][2]-cropboxes[idx][0]),
                (cropboxes[idx][3]-cropboxes[idx][1]),
                (cropboxes[idx][2]-cropboxes[idx][0]),
                (cropboxes[idx][3]-cropboxes[idx][1]),
            ],0)

            offset = tf.cast(tf.stack([
                cropboxes[idx][0], cropboxes[idx][1],
                cropboxes[idx][0], cropboxes[idx][1],
                ],0), tf.float32
            )

            # Calculate new bbox size
            boxes = (boxes - offset)/scale

            return crops[idx,:,:,:], boxes

        def nocrop(img, boxes):
            return img, boxes

        iw = self.optionsDict["dataSource"]["iw"]
        ih = self.optionsDict["dataSource"]["ih"]

        # =======================
        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        img, boxes = tf.cond(
            choice < 0.5,
            lambda: nocrop(img, boxes),
            lambda: cropimage(img, boxes, width=iw, height=ih)
        )

        return img, boxes, labels, jsonfile


    # ============================
    def _processAddNoise(self, img, boxes, labels, jsonfile, **kwargs):

        def addnoise(img):
            weight = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
            gnoise = tf.random.normal(
                shape=tf.shape(img),
                mean=self.optionsDict["augs"]["noise"]["mean"],
                stddev=self.optionsDict["augs"]["noise"]["stddev"],
                dtype=tf.float32
            )

            return tf.clip_by_value(tf.add(img, gnoise * weight), 0.0, 1.0)

        def nonoise(img):
            return img

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        img = tf.cond(
            choice < 0.5,
            lambda: addnoise(img),
            lambda: nonoise(img)
        )

        return img, boxes, labels, jsonfile

    # ============================
    def _processAugmentFlip(self, img, boxes, labels, jsonfile):

        # Flip
        def flip(img, boxes):
            img = tf.image.flip_left_right(img)
            boxes = tf.stack([
                1.0-boxes[:,2],boxes[:,1],
                1.0-boxes[:,0],boxes[:,3],
            ], 1)

            return img, boxes

        def noflip(img, boxes):
            return img, boxes

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        img, boxes = tf.cond(choice < 0.5,
            lambda: noflip(img, boxes),
            lambda: flip(img, boxes)
        )

        return img, boxes, labels, jsonfile

    # ============================
    def _processAugmentFlipVertically(self, img, boxes, labels, jsonfile):

        # Flip
        def flip(img, boxes):
            img = tf.image.flip_up_down(img)
            boxes = tf.stack([
                boxes[:,0],1.0-boxes[:,3],
                boxes[:,2],1.0-boxes[:,1],
            ], 1)

            return img, boxes

        def noflip(img, boxes):
            return img, boxes

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        img, boxes = tf.cond(choice < 0.5,
            lambda: noflip(img, boxes),
            lambda: flip(img, boxes)
        )

        return img, boxes, labels, jsonfile



    # ============================
    def _gaussianLabel(self, img, boxes, labels, jsonfile):
        """Returns Gaussian Heatmap
        """

        N = tf.shape(boxes)[0]

        # ===========================
        # Calculate class score [N,C]
        classScore = tf.one_hot(labels, depth=self.nc)

        # ===========================
        # KEYPOINTS
        # ===========================
        # Grid dimensions
        G = tf.expand_dims(tf.constant([self.ny-1, self.nx-1], dtype=tf.float32),0)

        # Calculate box centroids and best matching cell (ixyc) [N,2]
        p = tf.transpose(tf.stack((
            0.5 * (boxes[:, 1] + boxes[:, 3]),
            0.5 * (boxes[:, 0] + boxes[:, 2]),
        )))

        # Approximated cells
        inds = tf.cast(p*G, tf.int32)
        ptilde = tf.cast(inds, tf.float32)/G

        # ===========================
        # HEATMAP
        # ===========================
        # Calculate mesh  [H,W,1,2]
        axx, ayy = tf.meshgrid( tf.linspace(0,1,self.ny), tf.linspace(0,1,self.nx))
        ax = tf.stack([ayy,axx], axis=-1)
        ax = tf.expand_dims(tf.cast(ax, tf.float32),-2)


        # Gaussian Kernel Smearing [H,W,N]
        hm = tf.exp(
            tf.reduce_sum(-0.5* (tf.pow(
                tf.expand_dims(tf.expand_dims(ptilde ,0),0) - ax,
                2)/tf.pow(self.sigma, 2)), axis=-1)
        )

        # Class correction [H,W,N] x [N,C] = [H,W,C]
        hm = tf.matmul(hm, classScore)

        # ===========================
        # OBJECTSIZE & Correct position
        # ===========================
        # width & height [N,2]
        wh = tf.transpose(tf.stack((
            (boxes[:, 3] - boxes[:, 1]),
            (boxes[:, 2] - boxes[:, 0]),
        )))

        wh = tf.scatter_nd(
          indices=inds,
          updates=wh,
          shape=(self.nx,self.ny,2)
        )
   
        # position correction [N,2]
        pdelta = (p-ptilde)

        pdelta = tf.scatter_nd(
          indices=inds,
          updates=pdelta,
          shape=(self.nx,self.ny,2)
        )

        # Index heatmap
        idx = tf.scatter_nd(
          indices=inds,
          updates=tf.ones(shape=(N,1)),
          shape=(self.nx,self.ny,1)
        )
   
        return img, (hm, wh, pdelta, idx)

    # ============================


def postprocess(ylabel, pool_size=3, K=50):


    # [B,H,W,C], [B,H,W,2], [B,H,W,2]
    hm, wh, pdelta, inds = ylabel # tf.split(ylabel, [C, 2, 2], axis=-1)
    
    B, H, W, C = hm.shape

    # ======================================
    # All boxcoordinates
    # Grid [1,H,W,2]
    axx, ayy = tf.meshgrid( tf.linspace(0,1,H), tf.linspace(0,1,W))
    ax = tf.stack([ayy,axx], axis=-1)
    ax = tf.expand_dims(tf.cast(ax, tf.float32),0)

    # Box coordinates [B,H,W,4] [Y1,X1,Y2,X2]
    byx  = tf.concat([
            (ax + pdelta) - 0.5*wh,
            (ax + pdelta) + 0.5*wh,
        ],
        axis=-1
    )


    # [B,H,W,C,4]
    byxc = tf.tile(tf.expand_dims(byx, -2), [1,1,1,C,1] )

    # [B,HWC,4]
    byxc = tf.reshape(byxc, shape=(B, -1, 4))


   # print(tf.reduce_max(tf.abs(dx), axis=1))


    # ======================================
    # Filter out with MaxPool
    hmax = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=1, padding="same")(hm)
    keep = tf.cast(tf.equal(hm, hmax), tf.float32)
    hm =  hmax * keep # [B,H,W,C]

    # [B,HWC]
    score = tf.reshape(hm, shape=(B, -1))

    inds = tf.argsort(score, axis=1, direction='DESCENDING')
    inds = tf.slice(inds, [0, 0], [1, K]) 

    wh_ = tf.tile(tf.expand_dims(wh, -2), [1,1,1,C,1] )
    wh_ = tf.reshape(wh_, shape=(B, -1, 2))
    wh_ = tf.gather(params=wh_, indices=inds, batch_dims=1)

    pdelta_ = tf.tile(tf.expand_dims(pdelta, -2), [1,1,1,C,1] )
    pdelta_ = tf.reshape(pdelta_, shape=(B, -1, 2))
    pdelta_ = tf.gather(params=pdelta_, indices=inds, batch_dims=1)


    score = tf.gather(params=score, indices=inds, batch_dims=1)
    byxc = tf.gather(params=byxc, indices=inds, batch_dims=1)

    print("SCORE:", score)
    print(wh_)
    print(pdelta_)

    print(byxc)


    return byxc, hm




if __name__ == "__main__":

    classNames = ["face", "mask", "dummy"]
    datapath = "/data/projects/datasets/hands/test"


    ih, iw, ic = 256,256,3
    nx, ny = 32,32
    batchSize = 1

    dp = Datapipe(datapath, classNames)
    g = dp.create(nx, ny, iw, ih, ic, batchSize, shuffle_buffer_size=5000, nrepeat=1, minBoxSize=6, sigma=0.02)

    import matplotlib.pyplot as plt
    import cv2

    ctr = 0
    for x in g.take(4):
        print("=====================")

        #print(x[2])
      #  print(tf.reduce_min(x[1][...,5:], axis=[1,2]))

        byxc, keep = postprocess(x[1])

        #print(keep)
        # print("....")
        # print(x)
        # print(y)
        # print(ind)

        img = x[0][0,...].numpy()
        hm = cv2.resize(x[1][0][0,...].numpy(),(ih,iw))
        
        for k in range(byxc.shape[-2]):

            y1,x1,y2,x2 = (byxc.numpy()[0,k,:]*np.asarray([ih,iw,ih,iw])).astype(np.int32)
            print(y1,x1,y2,x2 )
            print(byxc.numpy()[0,k,:])
            img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)


        plt.imshow(tf.concat((img, hm, img+hm), axis=1), interpolation='none', aspect='equal')
        plt.savefig(f"./input_{ctr}.png")
        plt.close()


        fig,axs = plt.subplots(3)

        axs[0].imshow(x[1][0][0,...], vmin=0, vmax=1.0,  interpolation='none', aspect='equal')
        axs[1].imshow(x[1][1][0,...,0], vmin=0, vmax=0.05,  interpolation='none', aspect='equal')
        axs[2].imshow(tf.reduce_sum(keep[0,...,:3],axis=-1)*x[1][1][0,...,0], vmin=0, vmax=0.05,  interpolation='none', aspect='equal')

        # Major ticks
        axs[0].set_xticks(np.arange(0, nx, 1))
        axs[0].set_yticks(np.arange(0, ny, 1))

        # Labels for major ticks
      #  axs[0].set_xticklabels(np.arange(1, nx+1, 1))
      #  axs[0].set_yticklabels(np.arange(1, nx+1, 1))

        # Minor ticks
        axs[0].set_xticks(np.arange(-.5, nx, 1), minor=True)
        axs[0].set_yticks(np.arange(-.5, ny, 1), minor=True)

        axs[0].grid(which='minor', color='w', linestyle='-', linewidth=0.1)

        #axs[2].imshow(keep[0,...,:])
        plt.savefig(f"./scorr_{ctr}.png")
        plt.close()


        

        ctr +=1