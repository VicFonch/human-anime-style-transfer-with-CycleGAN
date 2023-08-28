from imports.common_imports import *

class DoubleConv(Layer):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        kernel_init: str = RandomNormal(stddev=0.02, seed=np.random.seed(42)),
        strides: int = 2,
        kernel_size: int = 4,
        residual: bool = False,
        relu_alpha = 0.2
    ):
        super(DoubleConv, self).__init__()
        
        self.residual = residual
        if not mid_channels:
            self.double_conv = Sequential([
                Conv2D(out_channels, kernel_size=kernel_size, strides=strides, 
                       padding='same', kernel_initializer=kernel_init),
                InstanceNormalization(axis=-1),
                LeakyReLU(alpha=relu_alpha)
            ])  
        else:
            self.double_conv = Sequential([
                Conv2D(mid_channels, kernel_size=kernel_size, strides=strides,
                        padding='same', kernel_initializer=kernel_init),
                InstanceNormalization(axis=-1),
                LeakyReLU(alpha=relu_alpha),
                # C256: 4x4 kernel Stride 2x2
                Conv2D(out_channels, kernel_size=kernel_size, strides=strides,
                        padding='same', kernel_initializer=kernel_init),
                InstanceNormalization(axis=-1),
                LeakyReLU(alpha=relu_alpha),
            ])
    
    def call(self, x):
        if self.residual:
            return tf.nn.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class DoubleConvTranspose(Layer):
        
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        strides: int = 2,
        kernel_size: int = 4,
        kernel_init: str = RandomNormal(stddev=0.02, seed=np.random.seed(42)),
        activation = 'relu',
        relu_alpha = 0.2
    ):
        super(DoubleConvTranspose, self).__init__()

        if not mid_channels:
            self.double_conv = Sequential([
                Conv2DTranspose(out_channels, kernel_size=kernel_size, strides=strides, 
                        padding='same', kernel_initializer=kernel_init),
                InstanceNormalization(axis=-1),
                Activation(activation)
            ])
        else:
            self.double_conv = Sequential([
                Conv2DTranspose(mid_channels, kernel_size=kernel_size, strides=strides,
                        padding='same', kernel_initializer=kernel_init),
                InstanceNormalization(axis=-1),
                Activation(activation),
                # C256: 4x4 kernel Stride 2x2
                Conv2DTranspose(out_channels, kernel_size=kernel_size, strides=strides,
                        padding='same', kernel_initializer=kernel_init),
                InstanceNormalization(axis=-1),
                Activation(activation),
            ])

    def call(self, x):
        return self.double_conv(x)

    
def Discriminator(input_shape=(256, 256, 3)):
    init = RandomNormal(stddev=0.02, seed=np.random.seed(42))
    input_image = Input(shape=input_shape)
    x = DoubleConv(in_channels=3,out_channels=64,strides=2)(input_image)
    x = DoubleConv(in_channels=64,out_channels=256,mid_channels=128, strides=2)(x)
    x = DoubleConv(in_channels=256,out_channels=512,strides=2)(x)
    x = DoubleConv(in_channels=512,out_channels=512,strides=1)(x)
    patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(x)
    model = Model(input_image, patch_out)

    bce = BinaryCrossentropy(from_logits=True)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=bce, optimizer=opt, loss_weights=[0.5])

    return model


def Generator(input_shape=(256, 256, 3), n_resnet=9):
    input_image = Input(shape=input_shape)
    x = DoubleConv(in_channels=3,out_channels=64,kernel_size=7,
                            strides = 1)(input_image)
    x = DoubleConv(in_channels=64,out_channels=128,kernel_size=3,
                            strides = 2)(x) 
    x = DoubleConv(in_channels=128,out_channels=256,kernel_size=3,
                            strides = 2)(x) 
    for _ in range(n_resnet):
        x = DoubleConv(in_channels=256,out_channels=256,mid_channels=256,kernel_size=3,
                            strides = 1,residual=True)(x)
    x = DoubleConvTranspose(in_channels=256,out_channels=65,mid_channels=128,kernel_size=3,
                            strides = 2)(x)
    out_image = DoubleConvTranspose(in_channels=65,out_channels=3,kernel_size=7,
                            strides = 1, activation='tanh')(x)
    model = Model(input_image, out_image)
    return model


def gauss_kernel(size=5, sigma=1.0):
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    return kernel

def conv_gauss(t_input, stride=1, k_size=5, sigma=1.6, repeats=1):
    t_kernel = tf.reshape(tf.constant(gauss_kernel(size=k_size, sigma=sigma), tf.float32),
                            [k_size, k_size, 1, 1])
    t_kernel3 = tf.concat([t_kernel]*t_input.get_shape()[3], axis=2)
    t_result = t_input
    for r in range(repeats):
        t_result = tf.nn.depthwise_conv2d(t_result, t_kernel3,
            strides=[1, stride, stride, 1], padding='SAME')
    return t_result

def make_laplacian_pyramid(t_img, max_levels):
    t_pyr = []
    current = t_img
    for level in range(max_levels):
        t_gauss = conv_gauss(current, stride=1, k_size=5, sigma=2.0)
        t_diff = current - t_gauss
        t_pyr.append(t_diff)
        current = tf.nn.avg_pool(t_gauss, [1,2,2,1], [1,2,2,1], 'VALID')
    t_pyr.append(current)
    return t_pyr

def laploss(t_img1, t_img2, max_levels=3):
    t_pyr1 = make_laplacian_pyramid(t_img1, max_levels)
    t_pyr2 = make_laplacian_pyramid(t_img2, max_levels)
    t_losses = [tf.norm(a-b,ord=1)/tf.size(a, out_type=tf.float32) for a,b in zip(t_pyr1, t_pyr2)]
    t_loss = tf.reduce_sum(t_losses)*tf.shape(t_img1, out_type=tf.float32)[0]
    return t_loss

def CycleModel(generator_A, generator_B, discriminator, image_shape=(256, 256, 3)):
    
    generator_A.trainable = True
	# mark discriminator and second generator as non-trainable
    discriminator.trainable = False
    generator_B.trainable = False

    input_gen = Input(shape=image_shape)
    gen1_out = generator_A(input_gen)
    output_d = discriminator(gen1_out)
    # identity loss
    input_id = Input(shape=image_shape)
    output_id = generator_A(input_id)
    # cycle loss - forward
    output_f = generator_B(gen1_out)
    # cycle loss - backward
    gen2_out = generator_B(input_id)
    output_b = generator_A(gen2_out)

    model = Model(
        [input_gen, input_id],
        [output_d, output_id, output_f, output_b]
    )

    lpyloss = lambda img1, img2: laploss(img1, img2)
    bce = BinaryCrossentropy(from_logits=True)

    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=[bce, lpyloss, lpyloss, lpyloss], 
                  loss_weights=[1, 5, 10, 10], optimizer=optimizer) 
    
    return model