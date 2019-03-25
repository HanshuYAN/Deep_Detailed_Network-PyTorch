# This tensorflow implementation of guided filter is from: H. Wu, et al., "Fast End-to-End Trainable Guided Filter", CPVR, 2018.

# Web: https://github.com/wuhuikai/DeepGuidedFilter

# import tensorflow as tf
import torch

def diff_x(input, r):
    assert len(input.shape) == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat((left, middle, right), dim=2)

    return output


def diff_y(input, r):
    assert len(input.shape) == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat((left, middle, right), dim=3)

    return output


def box_filter(x, r):
    assert len(x.shape) == 4

    return diff_y(torch.cumsum(diff_x(torch.cumsum(x, dim=2), r), dim=3), r)


def guided_filter(x, y, r, eps=1e-8, nhwc=False, nchw=True):
    assert len(x.shape) == 4 and len(y.shape) == 4

    # data format
    # if nhwc:
    #     x = tf.transpose(x, [0, 3, 1, 2])
    #     y = tf.transpose(y, [0, 3, 1, 2])
    assert nchw == True
    # shape check
    x_shape = x.shape
    y_shape = y.shape

    # x_shape = tf.shape(x)
    # y_shape = tf.shape(y)

    # assets = [tf.assert_equal(   x_shape[0],  y_shape[0]),
    #           tf.assert_equal(  x_shape[2:], y_shape[2:]),
    #           tf.assert_greater(x_shape[2:],   2 * r + 1),
    #           tf.Assert(tf.logical_or(tf.equal(x_shape[1], 1),
    #                                   tf.equal(x_shape[1], y_shape[1])), [x_shape, y_shape])]

    # with tf.control_dependencies(assets):
    #     x = tf.identity(x)

    # N
    cuda = torch.cuda.is_available()
    if cuda:
        N = box_filter(torch.ones((1, 1, x_shape[2], x_shape[3]), dtype=x.dtype).cuda(), r)

    # mean_x
    mean_x = box_filter(x, r) / N
    # mean_y
    mean_y = box_filter(y, r) / N
    # cov_xy
    cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
    # var_x
    var_x  = box_filter(x * x, r) / N - mean_x * mean_x

    # A
    A = cov_xy / (var_x + eps)
    # b
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N

    output = mean_A * x + mean_b

    # if nhwc:
    #     output = tf.transpose(output, [0, 2, 3, 1])

    return output