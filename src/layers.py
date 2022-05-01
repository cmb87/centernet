import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Lambda, MaxPool2D






class CenterNetPostprocessingLayer(tf.keras.Model):
    def __init__(self, nc, **kwargs):
        super(CenterNetPostprocessingLayer, self).__init__(**kwargs)
        self.nc = nc

    def call(self, x, training=False):

        x = Lambda( lambda x: tf.split(x, [self.nc, 2, 2], axis=-1), name="splitter")(x)

        # Heatmap branch
        y1 = Lambda( lambda x: tf.math.sigmoid(x), name="heatmap")(x[0])
        hmax = MaxPool2D(pool_size=3, strides=1, padding="same", name="heatmapNMS1")(y1)
        y4 = Lambda( lambda x: tf.cast(tf.equal(x[0], x[1]), tf.float32), name="mask")([y1,hmax])

        # Regression branch
        y2 = Lambda( lambda x: tf.math.sigmoid(x), "boxdimensions")(x[1])
        y3 = Lambda( lambda x: tf.math.tanh(x), "boxcorrection")(x[2])

        # Final output
        y = tf.concat((y1,y2,y3,y4), axis=-1)

        return y


class ImmediateSupvervision(tf.keras.Model):
    def __init__(self, nheatmaps, **kwargs):
        super(ImmediateSupvervision, self).__init__(**kwargs)
        self.nheatmaps = nheatmaps


    def build(self, input_shape):

        nf = input_shape[0][3]

        self.conv0 = Conv2D(nf, (1,1), padding="same", activation="relu")
     
        # Skip Connection
        self.conv2 = Conv2D(nf, (1,1),  padding="same", activation="relu")

        # Supervisied branch
        self.conv11 = Conv2D(self.nheatmaps, (1,1),  padding="same", activation="sigmoid")
        self.conv12 = Conv2D(nf, (1,1), padding="same", activation="relu")

        # Add everything
        self.addi = Add()


    def call(self, input_tensors, training=False):

        [xres, x] = input_tensors

        x = self.conv0(x)

        y = self.conv11(x)
        x1 = self.conv12(y)
        x2 = self.conv2(x)

        x = self.addi([xres, x1, x2])

        return [x, y]




class HourglassModule(tf.keras.Model):
    def __init__(self, nfilters, ndepths, **kwargs):
        super(HourglassModule, self).__init__(**kwargs)

        self.lowE = Residual(2*nfilters)
        self.lowD = Residual(nfilters)
        self.up = Residual(2*nfilters)

        self.p = Downsample(2)
        self.u = Upsample(2)

        if ndepths>1:
            self.hg = HourglassModule(nfilters=2*nfilters, ndepths=ndepths-1)
        else:
            self.hg = Residual(nfilters*2)


    def call(self, x, training=False):

        x = self.lowE(x)

        xfeat = self.p(x)
        xfeat = self.hg(xfeat)
        xfeat = self.u(xfeat)

        xresd = self.up(x)

        x = xresd + xfeat
        x = self.lowD(x)

        return x





class Residual(tf.keras.Model):
    def __init__(self, nf, dilation=(1,1),**kwargs):
        super(Residual, self).__init__(**kwargs)

        self.nf = nf
        self.dilation = dilation

    def build(self, input_shape):
        
        nf0 = input_shape[3]

        self.conv1 = Conv2D(int(0.5*self.nf), (1, 1), activation='relu', padding='same', dilation_rate=self.dilation)
        self.conv2 = Conv2D(int(0.5*self.nf), (3, 3), activation='relu', padding='same', dilation_rate=self.dilation)
        self.conv3 = Conv2D(self.nf, (1, 1), activation='relu', padding='same', dilation_rate=self.dilation)
        self.drop = Dropout(0.3)

        if nf0 == self.nf:
            self.need_skip = False
            self.skipConv = None
            print("None...")
        else:
            self.need_skip = True
            self.skipConv = Conv2D(self.nf, (1, 1), activation='relu', padding='same', dilation_rate=self.dilation)
            print(f"Inputshape {nf0} Outputshape {self.nf} ({input_shape})")
        

    def call(self, input_tensor, training=False):

        xred = input_tensor if not self.need_skip else self.skipConv(input_tensor)

        x = self.conv1(input_tensor, training=training)
        x = self.drop(x)
        x = self.conv2(x, training=training)
        x = self.drop(x)
        x = self.conv3(x, training=training)
        x += xred

        return x


class Upsample(tf.keras.Model):
    def __init__(self, kernelsize):
        super(Upsample, self).__init__(name="")
        self.conv1 = UpSampling2D(kernelsize)

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor, training=training)
        return x


class Downsample(tf.keras.Model):
    def __init__(self, maxpoolsize):
        super(Downsample, self).__init__(name="")
        self.pool1 = MaxPooling2D(maxpoolsize, padding="same")

    def call(self, input_tensor, training=False):
        x = self.pool1(input_tensor, training=training)
        return x
