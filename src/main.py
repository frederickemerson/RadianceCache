import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
from elsr.model import ELSR  # Import the elsr model

import tensorflow as tf
from tensorflow.keras import layers, models

'''
def build_mvideosr(scale_factor=4):
    """
    Build the MVideoSR model for 4x super-resolution.
    """
    inputs = layers.Input(shape=(None, None, 3))

    # First convolution layer
    x = layers.Conv2D(6, (3, 3), padding="same")(inputs)
    x = layers.PReLU()(x)

    # Residual blocks (3 layers)
    for _ in range(3):
        x = layers.Conv2D(6, (3, 3), padding="same")(x)
        x = layers.PReLU()(x)

    # Pixel shuffle for upscaling
    x = layers.Conv2D(scale_factor ** 2 * 3, (3, 3), padding="same")(x)
    outputs = tf.nn.depth_to_space(x, block_size=scale_factor)

    return models.Model(inputs, outputs)


def build_zxvip(scale_factor=4):
    """
    Build the ZX VIP model for 4x super-resolution.
    """
    inputs = layers.Input(shape=(None, None, 3))

    # Re-parametrization Edge-Oriented Convolution Block
    def rcbsr_block(x):
        branch1 = layers.Conv2D(6, (1, 1), padding="same")(x)
        branch2 = layers.Conv2D(6, (3, 3), padding="same")(x)
        branch3 = layers.Conv2D(6, (5, 5), padding="same")(x)
        return layers.add([branch1, branch2, branch3])

    # Apply the RCBSR block
    x = rcbsr_block(inputs)

    # Pixel shuffle for upscaling
    x = layers.Conv2D(scale_factor**2 * 3, (3, 3), padding="same")(x)
    outputs = tf.nn.depth_to_space(x, block_size=scale_factor)

    return models.Model(inputs, outputs)
'''

def main():
    global model
    checkpoint = utility.checkpoint(args)
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None

        # Handle elsr integration
        if args.use_elsr:
            checkpoint.write_log("Integrating elsr into NSRD...")
            if not args.elsr_path or not os.path.exists(args.elsr_path):
                from elsr.training import train_elsr_main
                args.elsr_path = train_elsr_main(args)

        t = Trainer(args, loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

if __name__ == '__main__':
    main()
