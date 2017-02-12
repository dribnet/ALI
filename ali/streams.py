"""Functions for creating data streams."""
from fuel.datasets import CIFAR10, SVHN, CelebA
from fuel.datasets.toy import Spiral
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import ForceFloatX

from .datasets import TinyILSVRC2012, GaussianMixture
from plat.fuel_helper import create_custom_streams

def create_svhn_data_streams(batch_size, monitoring_batch_size, rng=None):
    train_set = SVHN(2, ('extra',), sources=('features',))
    valid_set = SVHN(2, ('train',), sources=('features',))
    main_loop_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size, rng=rng))
    train_monitor_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    valid_monitor_stream = DataStream.default_stream(
        valid_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    return main_loop_stream, train_monitor_stream, valid_monitor_stream


def create_cifar10_data_streams(batch_size, monitoring_batch_size, rng=None):
    train_set = CIFAR10(
        ('train',), sources=('features',), subset=slice(0, 45000))
    valid_set = CIFAR10(
        ('train',), sources=('features',), subset=slice(45000, 50000))
    main_loop_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size, rng=rng))
    train_monitor_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    valid_monitor_stream = DataStream.default_stream(
        valid_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    return main_loop_stream, train_monitor_stream, valid_monitor_stream


def create_celeba_data_streams(batch_size, monitoring_batch_size,
                               sources=('features', ), rng=None):
    train_set = CelebA('64', ('train',), sources=sources)
    valid_set = CelebA('64', ('valid',), sources=sources)
    main_loop_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size, rng=rng))
    train_monitor_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            5000, monitoring_batch_size, rng=rng))
    valid_monitor_stream = DataStream.default_stream(
        dataset=valid_set,
        iteration_scheme=SequentialScheme(
            5000, monitoring_batch_size))
    return main_loop_stream, train_monitor_stream, valid_monitor_stream


def create_tiny_imagenet_data_streams(batch_size, monitoring_batch_size,
                                      rng=None):
    train_set = TinyILSVRC2012(('train',), sources=('features',))
    valid_set = TinyILSVRC2012(('valid',), sources=('features',))
    main_loop_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size, rng=rng))
    train_monitor_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            4096, monitoring_batch_size, rng=rng))
    valid_monitor_stream = DataStream.default_stream(
        valid_set,
        iteration_scheme=ShuffledScheme(
            4096, monitoring_batch_size, rng=rng))
    return main_loop_stream, train_monitor_stream, valid_monitor_stream


def create_spiral_data_streams(batch_size, monitoring_batch_size, rng=None,
                               num_examples=100000, classes=1, cycles=2,
                               noise=0.1):
    train_set = Spiral(num_examples=num_examples, classes=classes,
                       cycles=cycles, noise=noise, sources=('features',))

    valid_set = Spiral(num_examples=num_examples, classes=classes,
                       cycles=cycles, noise=noise, sources=('features',))

    main_loop_stream = DataStream.default_stream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size=batch_size, rng=rng))

    train_monitor_stream = DataStream.default_stream(
        train_set, iteration_scheme=ShuffledScheme(5000, batch_size, rng=rng))

    valid_monitor_stream = DataStream.default_stream(
        valid_set, iteration_scheme=ShuffledScheme(5000, batch_size, rng=rng))

    return main_loop_stream, train_monitor_stream, valid_monitor_stream


def create_gaussian_mixture_data_streams(batch_size, monitoring_batch_size,
                                         means=None, variances=None, priors=None,
                                         rng=None, num_examples=100000,
                                         sources=('features', )):
    train_set = GaussianMixture(num_examples=num_examples, means=means,
                                variances=variances, priors=priors,
                                rng=rng, sources=sources)

    valid_set = GaussianMixture(num_examples=num_examples,
                                means=means, variances=variances,
                                priors=priors, rng=rng, sources=sources)

    main_loop_stream = DataStream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size=batch_size, rng=rng))

    train_monitor_stream = DataStream(
        train_set, iteration_scheme=ShuffledScheme(5000, batch_size, rng=rng))

    valid_monitor_stream = DataStream(
        valid_set, iteration_scheme=ShuffledScheme(5000, batch_size, rng=rng))

    return main_loop_stream, train_monitor_stream, valid_monitor_stream

def celeba_128_stream(batch_size, monitoring_batch_size,
                               sources=('features', ), rng=None):

    streams = create_custom_streams(filename="celeba_dlib2_128",
                                    training_batch_size=batch_size,
                                    monitoring_batch_size=monitoring_batch_size,
                                    include_targets=True,
                                    color_convert=False,
                                    split_names=("train","valid","test"))

    main_loop_stream, train_monitor_stream, valid_monitor_stream = streams[:3]

    return main_loop_stream, train_monitor_stream, valid_monitor_stream;

def celeba_64_stream(batch_size, monitoring_batch_size,
                               sources=('features', ), rng=None):

    streams = create_custom_streams(filename="celeba_dlib2_64",
                                    training_batch_size=batch_size,
                                    monitoring_batch_size=monitoring_batch_size,
                                    include_targets=True,
                                    color_convert=False,
                                    split_names=("train","valid","test"))

    main_loop_stream, train_monitor_stream, valid_monitor_stream = streams[:3]

    return main_loop_stream, train_monitor_stream, valid_monitor_stream;

def load_generic_stream(filename, split_names, batch_size, monitoring_batch_size,
                               sources=('features', ), rng=None):

    streams = create_custom_streams(filename=filename,
                                    training_batch_size=batch_size,
                                    monitoring_batch_size=monitoring_batch_size,
                                    include_targets=True,
                                    color_convert=False,
                                    split_names=split_names)

    main_loop_stream, train_monitor_stream, valid_monitor_stream = streams[:3]
    main_loop_stream = ForceFloatX(main_loop_stream)
    train_monitor_stream = ForceFloatX(train_monitor_stream)
    valid_monitor_stream = ForceFloatX(valid_monitor_stream)

    return main_loop_stream, train_monitor_stream, valid_monitor_stream;
