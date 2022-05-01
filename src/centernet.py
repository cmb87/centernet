import os
import sys
from matplotlib.pyplot import axis
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, Lambda, MaxPool2D, Reshape
from layers import Residual, Downsample, Upsample, HourglassModule, CenterNetPostprocessingLayer
from datapipe import Datapipe



nfeat = 32
learnrate = 1e-5

classNames = ["face", "mask", "dummy"]
datapath = "/data/projects/datasets/hands/train"

ih, iw, ic = 256,256,3
nx, ny = 128,128
nc = len(classNames)
batchSize = 10



# ========= Datapipe =================
dp = Datapipe(datapath, classNames)
g = dp.create(nx, ny, iw, ih, ic, batchSize, shuffle_buffer_size=5000, nrepeat=1, minBoxSize=6, sigma=0.02)


# ====================================================
i = tf.keras.layers.Input((ih,iw,ic), name="rgb")

# ========= Entry Layers =================
x0 = Conv2D(nfeat, (7,7), name="entry01", padding="same", activation="relu")(i)
x0 = Residual(nfeat, name="entry02")(x0)

# ========= First Hourglas =================
x1 = HourglassModule(nfilters=32, ndepths=4, name="hourglass1")(x0)
#[x1, y1] = ImmediateSupvervision(nheatmaps, name="imsuper1")([x0, x1])

# ========= Second Hourglas =================
#x2 = HourglassModule(nfilters=32, ndepths=2, name="hourglass2")(x1)
#[x2, y2] = ImmediateSupvervision(nheatmaps, name="imsuper2")([x1, x2])

# ========= Final prediction =================
x = Conv2D(nfeat, (3,3), strides=2, name="strider", padding="same", activation="relu")(x1)

x = Conv2D(nc+4, (1,1), name="postprocess", padding="same", activation="relu")(x)
y = CenterNetPostprocessingLayer(nc=nc)(x)


# ========= The model =================
model = tf.keras.Model(inputs=[i], outputs=[y])

print(model.summary())
print(model.outputs)


# ========= The loss =================


def centerNetLoss(ytrue, ypred):

   # K = tf.shape(ytrue)[-1]
   # C = K - 5
    
    C = nc 

    hmTrue, whTrue, pdeltaTrue, indsTrue = tf.split(ytrue, [C, 2, 2, 1], axis=-1)
    hmPred, whPred, pdeltaPred, indsPred = tf.split(ypred, [C, 2, 2, 3], axis=-1)

    print("----")
    print(hmTrue.shape, hmPred.shape)
    print(whTrue.shape, whTrue.shape)
    print(pdeltaTrue.shape, pdeltaPred.shape)
    print(indsTrue.shape, indsPred.shape)



    lossHm = tf.keras.losses.binary_focal_crossentropy(hmTrue, hmPred, from_logits=False, gamma=2)

    lossPdelta = tf.reduce_sum(indsTrue*(tf.math.abs(pdeltaTrue-pdeltaPred)), axis=[1,2,3])
    lossWh = tf.reduce_sum(indsTrue*(tf.math.abs(whTrue-whPred)), axis=[1,2,3])


    # loss0 = tf.reduce_mean(loss0,axis=[1,2])
    # loss1 = tf.reduce_mean(loss1,axis=[1,2])
    
    # c = tf.cast(tf.math.greater(loss0, loss1), tf.float32)
    # loss = (1-c)*loss0 + c*loss1


    return lossHm  #+ 0.5*(lossPdelta+ lossWh)

# ============================================
# Training
# ============================================

tfbcb = tf.keras.callbacks.TensorBoard(
    log_dir="./tblogs", histogram_freq=0, write_graph=True,
    write_images=True, update_freq='batch',
    profile_batch=2, embeddings_freq=0, embeddings_metadata=None
)

estcb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
)

mcpcb = tf.keras.callbacks.ModelCheckpoint(
    os.path.join('weights_cpk.h5'), monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch',
)


if os.path.isfile('weights_cpk.h5'): 
    model.load_weights('weights_cpk.h5')
    print("weights loaded")


model.compile(
    loss=centerNetLoss,
    optimizer=tf.keras.optimizers.Adam(learnrate)
)

model.fit(
    g, epochs=300,
    callbacks = [tfbcb, mcpcb, estcb],
  #  validation_data=dste
)

model.save_weights("weights.h5")




