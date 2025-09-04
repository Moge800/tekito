from keras.api import layers, models


class BuildTinyUNet:
    def __init__(self, input_shape=(224, 224, 1), base_filters=16):
        self.input_shape = input_shape
        self.base_filters = base_filters
        self.model = self.build_model()

    def conv_block(self, x, filters, name_prefix):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name_prefix}_conv1")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name_prefix}_conv2")(x)
        return x

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape, name="input")

        # Encoder
        c1 = self.conv_block(inputs, self.base_filters, "enc1")
        p1 = layers.MaxPooling2D((2, 2), name="pool1")(c1)

        c2 = self.conv_block(p1, self.base_filters * 2, "enc2")
        p2 = layers.MaxPooling2D((2, 2), name="pool2")(c2)

        # Bottleneck
        b = self.conv_block(p2, self.base_filters * 4, "bottleneck")

        # Decoder
        u2 = layers.UpSampling2D((2, 2), name="up2")(b)
        concat2 = layers.Concatenate(name="concat2")([u2, c2])
        c3 = self.conv_block(concat2, self.base_filters * 2, "dec2")

        u1 = layers.UpSampling2D((2, 2), name="up1")(c3)
        concat1 = layers.Concatenate(name="concat1")([u1, c1])
        c4 = self.conv_block(concat1, self.base_filters, "dec1")

        outputs = layers.Conv2D(1, 1, activation="sigmoid", name="output")(c4)

        model = models.Model(inputs, outputs, name="TinyUNet")
        return model


# Instantiate and summarize the model
tiny_unet_model = BuildTinyUNet().model
tiny_unet_model.summary()
tiny_unet_model.save("./tiny_u-net.keras")
