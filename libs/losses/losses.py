#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf


# def l1_smooth_losses(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights=None, sigma=1.0, dim=[1]):
#     sigma_2 = sigma ** 2
#     box_diff = bbox_pred - bbox_targets
#     in_box_diff = bbox_inside_weights * tf.reduce_sum(box_diff)
#     abs_in_box_diff = tf.abs(in_box_diff)
#     smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
#     in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
#                   + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
#     # out_loss_box = bbox_outside_weights * in_loss_box
#     # loss_box = tf.reduce_mean(tf.reduce_sum(
#     #     out_loss_box,
#     #     axis=dim
#     # ))
#     if bbox_outside_weights is None:
#         out_loss_box = in_loss_box
#     else:
#         out_loss_box = bbox_outside_weights * in_loss_box
#     loss_box = tf.reduce_mean(tf.reduce_sum(
#         out_loss_box,
#         axis=dim
#     ))
#     return loss_box

def l1_smooth_losses(predict_boxes, gtboxes, object_weights, classes_weights=None):
    '''
  :param predict_boxes: [minibatch_size, -1]
  :param gtboxes: [minibatch_size, -1]
  :param object_weights: [minibatch_size, ]. 1.0 represent object, 0.0 represent others(ignored or background)
  :return:
  '''
    diff = predict_boxes - gtboxes
    abs_diff = tf.cast(tf.abs(diff), tf.float32)
    # smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_diff, 1. )))

    if classes_weights is None:
        '''
    first_stage:
    predict_boxes :[minibatch_size, 4]
    gtboxes: [minibatchs_size, 4]
    '''
        anchorwise_smooth_l1norm = tf.reduce_sum(
            tf.where(tf.less(abs_diff, 1), 0.5 * tf.square(abs_diff), abs_diff - 0.5), axis=1) * object_weights
        # anchorwise_smooth_l1norm = tf.reduce_sum(
        #     tf.pow(abs_diff, 2) * (1 / 2.) * smoothL1_sign \
        #     + (abs_diff - (0.5 / 1)) * (1. - smoothL1_sign),
        #     axis=1) * object_weights
    else:
        '''
    fast_rcnn:
    predict_boxes: [minibatch_size, 4*num_classes]
    gtboxes: [minibatch_size, 4*num_classes]
    classes_weights : [minibatch_size, 4*num_classes]
    '''
        # anchorwise_smooth_l1norm = tf.reduce_sum(
        #     tf.pow(abs_diff, 2) * (1 / 2.) * classes_weights * smoothL1_sign \
        #     + (abs_diff - (0.5 / 1)) * classes_weights * (1. - smoothL1_sign),
        #     axis=1) * object_weights
        anchorwise_smooth_l1norm = tf.reduce_sum(tf.where(tf.less(abs_diff, 1), 0.5 * tf.square(
            abs_diff) * classes_weights, (abs_diff - 0.5) * classes_weights), axis=1) * object_weights
    return tf.reduce_mean(anchorwise_smooth_l1norm, axis=0) # reduce mean


def weighted_softmax_cross_entropy_loss(predictions, labels, weights):
    '''
  :param predictions:
  :param labels:
  :param weights: [N, ] 1 -> should be sampled , 0-> not should be sampled
  :return:
  '''
    per_row_cross_ent = tf.nn.softmax_cross_entropy_with_logits(logits=predictions,
                                                                labels=labels)

    weighted_cross_ent = tf.reduce_sum(per_row_cross_ent * weights)
    return weighted_cross_ent / tf.reduce_sum(weights)


def test_smoothl1():
    predict_boxes = tf.constant([[1, 1, 2, 2],
                                 [2, 2, 2, 2],
                                 [3, 3, 3, 3]])
    gtboxes = tf.constant([[1, 1, 1, 1],
                           [2, 1, 1, 1],
                           [3, 3, 2, 1]])

    classes_weights = tf.constant([[0.8], [0.9], [0.8]])
    loss = l1_smooth_losses(predict_boxes, gtboxes, [1, 1, 1], classes_weights=classes_weights)

    with tf.Session() as sess:
        print(sess.run(loss))


if __name__ == '__main__':
    test_smoothl1()
