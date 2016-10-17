import argparse
import logging

from blocks.algorithms import Adam
from blocks.bricks import LeakyRectifier, Logistic
from blocks.bricks.conv import ConvolutionalSequence
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.graph.bn import (batch_normalization,
                             get_batch_normalization_updates)
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import INPUT
from blocks.serialization import load
from theano import tensor

from ali.algorithms import ali_algorithm
from ali.bricks import (ALI, GaussianConditional, DeterministicConditional,
                        XZJointDiscriminator)
from ali.streams import create_celeba_data_streams
from ali.utils import get_log_odds, conv_brick, conv_transpose_brick, bn_brick

from ali.interface import AliModel
from plat.training.samplecheckpoint import SampleCheckpoint
from plat.fuel_helper import create_custom_streams

NUM_CHANNELS = 3
GAUSSIAN_INIT = IsotropicGaussian(std=0.01)
ZERO_INIT = Constant(0)
LEARNING_RATE = 1e-4
BETA1 = 0.5
LEAK = 0.02


def create_model_brick(model_stream, image_size, z_dim):

    if image_size == 64:
        encoder_layers = [
            conv_brick(2, 1, 64), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(7, 2, 128), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(5, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(7, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(4, 1, 512), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(1, 1, 2 * z_dim)]

        decoder_layers = [
            conv_transpose_brick(4, 1, 512), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_transpose_brick(7, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_transpose_brick(5, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_transpose_brick(7, 2, 128), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_transpose_brick(2, 1, 64), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(1, 1, NUM_CHANNELS), Logistic()]

        x_disc_layers = [
            conv_brick(2, 1, 64), LeakyRectifier(leak=LEAK),
            conv_brick(7, 2, 128), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(5, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(7, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(4, 1, 512), bn_brick(), LeakyRectifier(leak=LEAK)]

    elif image_size == 128:
        encoder_layers = [
            conv_brick(4, 1, 64), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(7, 2, 128), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(6, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(6, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(6, 2, 512), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(4, 1, 1024), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(1, 1, 2 * z_dim)]

        decoder_layers = [
            conv_transpose_brick(4, 1, 1024), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_transpose_brick(6, 2, 512), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_transpose_brick(6, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_transpose_brick(6, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_transpose_brick(7, 2, 128), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_transpose_brick(4, 1, 64), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(1, 1, NUM_CHANNELS), Logistic()]

        x_disc_layers = [
            conv_brick(4, 1, 64), LeakyRectifier(leak=LEAK),
            conv_brick(7, 2, 128), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(6, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(6, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(6, 2, 512), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(4, 1, 1024), bn_brick(), LeakyRectifier(leak=LEAK)]

    elif image_size == 256:
        encoder_layers = [
            conv_brick(4, 1, 64), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(7, 2, 128), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(6, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(6, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(6, 2, 512), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(6, 2, 512), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(4, 1, 1024), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(1, 1, 2 * z_dim)]

        decoder_layers = [
            conv_transpose_brick(4, 1, 1024), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_transpose_brick(6, 2, 512), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_transpose_brick(6, 2, 512), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_transpose_brick(6, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_transpose_brick(6, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_transpose_brick(7, 2, 128), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_transpose_brick(4, 1, 64), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(1, 1, NUM_CHANNELS), Logistic()]

        x_disc_layers = [
            conv_brick(4, 1, 64), LeakyRectifier(leak=LEAK),
            conv_brick(7, 2, 128), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(6, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(6, 2, 256), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(6, 2, 512), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(6, 2, 512), bn_brick(), LeakyRectifier(leak=LEAK),
            conv_brick(4, 1, 1024), bn_brick(), LeakyRectifier(leak=LEAK)]

    print("Building network with {} enc, {} dec, and {} x_disc layers".format(
        len(encoder_layers), len(decoder_layers), len(x_disc_layers)))

    encoder_mapping = ConvolutionalSequence(
        layers=encoder_layers, num_channels=NUM_CHANNELS, image_size=(image_size, image_size),
        use_bias=False, name='encoder_mapping')
    encoder = GaussianConditional(encoder_mapping, name='encoder')

    decoder_mapping = ConvolutionalSequence(
        layers=decoder_layers, num_channels=z_dim, image_size=(1, 1), use_bias=False,
        name='decoder_mapping')
    decoder = DeterministicConditional(decoder_mapping, name='decoder')

    x_discriminator = ConvolutionalSequence(
        layers=x_disc_layers, num_channels=NUM_CHANNELS, image_size=(image_size, image_size),
        use_bias=False, name='x_discriminator')
    x_discriminator.push_allocation_config()

    layers = [
        conv_brick(1, 1, 1024), LeakyRectifier(leak=LEAK),
        conv_brick(1, 1, 1024), LeakyRectifier(leak=LEAK)]
    z_discriminator = ConvolutionalSequence(
        layers=layers, num_channels=z_dim, image_size=(1, 1), use_bias=False,
        name='z_discriminator')
    z_discriminator.push_allocation_config()

    layers = [
        conv_brick(1, 1, 2048), LeakyRectifier(leak=LEAK),
        conv_brick(1, 1, 2048), LeakyRectifier(leak=LEAK),
        conv_brick(1, 1, 1)]
    joint_discriminator = ConvolutionalSequence(
        layers=layers,
        num_channels=(x_discriminator.get_dim('output')[0] +
                      z_discriminator.get_dim('output')[0]),
        image_size=(1, 1),
        name='joint_discriminator')

    discriminator = XZJointDiscriminator(
        x_discriminator, z_discriminator, joint_discriminator,
        name='discriminator')

    ali = ALI(encoder, decoder, discriminator,
              weights_init=GAUSSIAN_INIT, biases_init=ZERO_INIT,
              name='ali')
    ali.push_allocation_config()
    encoder_mapping.layers[-1].use_bias = True
    encoder_mapping.layers[-1].tied_biases = False
    decoder_mapping.layers[-2].use_bias = True
    decoder_mapping.layers[-2].tied_biases = False
    x_discriminator.layers[0].use_bias = True
    x_discriminator.layers[0].tied_biases = True
    ali.initialize()
    raw_marginals, = next(model_stream.get_epoch_iterator())
    b_value = get_log_odds(raw_marginals)
    decoder_mapping.layers[-2].b.set_value(b_value)

    return ali


def create_models(model_stream, image_size, z_dim, oldmodel=None):
    if oldmodel is None:
        ali = create_model_brick(model_stream, image_size, z_dim)
    else:
        ali = oldmodel

    x = tensor.tensor4('features')
    z = ali.theano_rng.normal(size=(x.shape[0], z_dim, 1, 1))

    def _create_model(with_dropout):
        cg = ComputationGraph(ali.compute_losses(x, z))
        if with_dropout:
            inputs = VariableFilter(
                bricks=([ali.discriminator.x_discriminator.layers[0]] +
                        ali.discriminator.x_discriminator.layers[2::3] +
                        ali.discriminator.z_discriminator.layers[::2] +
                        ali.discriminator.joint_discriminator.layers[::2]),
                roles=[INPUT])(cg.variables)
            cg = apply_dropout(cg, inputs, 0.2)
        return Model(cg.outputs)

    model = _create_model(with_dropout=False)

    with batch_normalization(ali):
        bn_model = _create_model(with_dropout=True)

    pop_updates = list(
        set(get_batch_normalization_updates(bn_model, allow_duplicates=True)))
    bn_updates = [(p, m * 0.05 + p * 0.95) for p, m in pop_updates]

    return model, bn_model, bn_updates


def create_main_loop(save_path, subdir, dataset, splits, color_convert,
        batch_size, monitor_every, checkpoint_every, num_epochs,
        image_size, z_dim, oldmodel):

    if dataset is None:
        streams = create_celeba_data_streams(batch_size, batch_size)
        model_stream = create_celeba_data_streams(500, 500)[0]
    else:
        streams = create_custom_streams(filename=dataset,
                                        training_batch_size=batch_size,
                                        monitoring_batch_size=batch_size,
                                        include_targets=False,
                                        color_convert=color_convert,
                                        split_names=splits)
        model_stream = create_custom_streams(filename=dataset,
                                        training_batch_size=500,
                                        monitoring_batch_size=500,
                                        include_targets=False,
                                        color_convert=color_convert,
                                        split_names=splits)[0]

    main_loop_stream, train_monitor_stream, valid_monitor_stream = streams[:3]

    old_model = None
    if oldmodel is not None:
        print("Initializing parameters with old model {}".format(oldmodel))
        with open(oldmodel, 'rb') as src:
            old_main_loop = load(src)
            old_model, = old_main_loop.model.top_bricks

    model, bn_model, bn_updates = create_models(model_stream, image_size, z_dim, old_model)
    ali, = bn_model.top_bricks
    discriminator_loss, generator_loss = bn_model.outputs

    step_rule = Adam(learning_rate=LEARNING_RATE, beta1=BETA1)
    algorithm = ali_algorithm(discriminator_loss, ali.discriminator_parameters,
                              step_rule, generator_loss,
                              ali.generator_parameters, step_rule)
    algorithm.add_updates(bn_updates)

    bn_monitored_variables = (
        [v for v in bn_model.auxiliary_variables if 'norm' not in v.name] +
        bn_model.outputs)
    monitored_variables = (
        [v for v in model.auxiliary_variables if 'norm' not in v.name] +
        model.outputs)
    train_monitoring = DataStreamMonitoring(
        bn_monitored_variables, train_monitor_stream, prefix="train",
        updates=bn_updates, after_epoch=False, before_first_epoch=False,
        every_n_epochs=monitor_every)
    valid_monitoring = DataStreamMonitoring(
        monitored_variables, valid_monitor_stream, prefix="valid",
        after_epoch=False, before_first_epoch=False,
        every_n_epochs=monitor_every)
    checkpoint = Checkpoint(save_path, every_n_epochs=checkpoint_every,
        before_training=True, after_epoch=True, after_training=True,
        use_cpickle=True)
    sampling_checkpoint =  SampleCheckpoint(interface=AliModel, z_dim=z_dim,
        image_size=(image_size, image_size), channels=NUM_CHANNELS,
        dataset=dataset, split=splits[1], save_subdir=subdir,
        before_training=True, after_epoch=True)

    extensions = [
        Timing(),
        FinishAfter(after_n_epochs=num_epochs),
        checkpoint,
        sampling_checkpoint,
        train_monitoring,
        valid_monitoring,
        Printing(),
        ProgressBar(),
    ]

    main_loop = MainLoop(model=bn_model, data_stream=main_loop_stream,
                         algorithm=algorithm, extensions=extensions)

    return main_loop


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train ALI on CelebA")
    parser.add_argument('--model', dest='model', type=str,
                        default="ali_celeba.zip", help="Model to save")
    parser.add_argument("--subdir", dest='subdir', type=str, default="output",
                        help="Subdirectory for output files (images)")
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Dataset for training.")
    parser.add_argument('--splits', dest='splits', default="train,valid,test",
                        help="train/valid/test dataset split names")
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of (offset) images")
    parser.add_argument('--color-convert', dest='color_convert',
                        default=False, action='store_true',
                        help="Convert source dataset to color from grayscale.")
    parser.add_argument("--batch-size", type=int, dest="batch_size",
                        default=100, help="Size of each mini-batch")
    parser.add_argument("--monitor-every", type=int, dest="monitor_every",
                        default=4, help="Frequency in epochs for monitoring")
    parser.add_argument("--checkpoint-every", type=int,
                        dest="checkpoint_every", default=1,
                        help="Frequency in epochs for checkpointing")
    parser.add_argument("--num-epochs", type=int, dest="num_epochs",
                        default=123, help="Stop training after num-epochs.")
    parser.add_argument("--z-dim", type=int, dest="z_dim",
                        default=256, help="Z-vector dimension")
    parser.add_argument("--oldmodel", type=str, default=None,
                        help="Use a model file created by a previous run as\
                        a starting point for parameters")
    args = parser.parse_args()
    splits = args.splits.split(",")
    create_main_loop(args.model, args.subdir, args.dataset, splits,
        args.color_convert, args.batch_size, args.monitor_every,
        args.checkpoint_every, args.num_epochs, args.image_size,
        args.z_dim, args.oldmodel).run()
