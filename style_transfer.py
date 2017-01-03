#!/usr/bin/python
# Mohamed K. Eid (mohamedkeid@gmail.com)

import argparse
import custom_vgg19 as vgg19
import numpy as np
import tensorflow as tf
import utils
from scipy.misc import toimage

# Model Hyper Params
content_layer = 'conv4_2'
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
epochs = 1000
learning_rate = 0.01
total_variation_smoothing = 1.5
norm_term = 6.0

# Loss term weights
content_weight = .001
style_weight = 1.0
norm_weight = 0.001
tv_weight = 0.01

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input", default='photo.jpg', help="path to the input image you would like to apply the style to")
parser.add_argument("--style", default='art.jpg', help="path to the image to be used as the style for the input image")
parser.add_argument("--output", default='./', help="path to where the image with the applied style will be created")
args = parser.parse_args()

# Assign image paths from the arg parsing
input_path = args.input
style_path = args.style
out_path = args.output


# Given an activated filter maps of any particular layer, return its respected gram matrix
def convert_to_gram(filter_maps):
    # Get the dimensions of the filter maps to reshape them into two dimenions
    dimension = filter_maps.get_shape().as_list()
    reshaped_maps = tf.reshape(filter_maps, [dimension[1] * dimension[2], dimension[3]])

    # Compute the inner product to get the gram matrix
    if dimension[1] * dimension[2] > dimension[3]:
        return tf.matmul(reshaped_maps, reshaped_maps, transpose_a=True)
    else:
        return tf.matmul(reshaped_maps, reshaped_maps, transpose_b=True)


# Compute the content loss given a variable image (x) and a content image (c)
def get_content_loss(x, c):
    with tf.name_scope('get_content_loss'):
        # Get the activated VGG feature maps
        content_noise_out = getattr(x, content_layer)
        content_photo_out = getattr(c, content_layer)

        # Compute and return the content loss term as the mean squared error
        sum_of_squared_diffs = tf.reduce_sum(tf.square(tf.sub(content_noise_out, content_photo_out)))
        return 0.5 * sum_of_squared_diffs


# Compute style loss given a variable image (x) and a style image (s)
def get_style_loss(x, s):
    with tf.name_scope('get_style_loss'):
        style_layer_losses = [get_style_loss_for_layer(x, s, l) for l in style_layers]
        style_weights = tf.constant([0.2] * len(style_layer_losses), tf.float32)
        weighted_layer_losses = tf.mul(style_weights, tf.convert_to_tensor(style_layer_losses))
        return tf.reduce_sum(weighted_layer_losses)


# Compute the norm of a given image (x)
def get_norm(x):
    summed_terms = tf.reduce_sum(tf.pow(x, norm_term))
    return tf.pow(summed_terms, 1.0 / norm_term)


# Compute style loss for a layer (l) given the variable image (x) and the style image (s)
def get_style_loss_for_layer(x, s, l):
    with tf.name_scope('get_style_loss_for_layer'):
        # Compute gram matrices using the activated filter maps of the art and generated images
        x_layer_maps = getattr(x, l)
        s_layer_maps = getattr(s, l)
        x_layer_gram = convert_to_gram(x_layer_maps)
        s_layer_gram = convert_to_gram(s_layer_maps)

        # Make sure the feature map dimensions are the same
        assert_equal_shapes = tf.assert_equal(x_layer_maps.get_shape(), s_layer_maps.get_shape())
        with tf.control_dependencies([assert_equal_shapes]):
            # Compute the gram loss using the gram matrices
            style_layer_shape = x_layer_maps.get_shape().as_list()
            style_layer_elements = ((style_layer_shape[1] * style_layer_shape[2]) ** 2) * (style_layer_shape[3] ** 2)
            style_fraction = 1.0 / (4.0 * style_layer_elements)
            gram_loss = tf.reduce_sum(tf.square(x_layer_gram - s_layer_gram))

            # Compute the end layer loss as the weighted gram loss
            return style_fraction * gram_loss


# Compute total variation regularization loss term given a variable image (x) and its shape
def get_total_variation(x, shape):
    with tf.name_scope('get_total_variation'):
        # Get the dimensions of the variable image
        height = shape[1]
        width = shape[2]

        # Disjoin the variable image and evaluate the total variation
        x_cropped = x[:, :height - 1, :width - 1, :]
        left_term = tf.square(x[:, 1:, :width - 1, :] - x_cropped)
        right_term = tf.square(x[:, :height - 1, 1:, :] - x_cropped)
        smoothed_summed_terms = tf.pow(tf.add(left_term, right_term), tf.div(total_variation_smoothing, 2))
        return tf.pow(tf.reduce_sum(smoothed_summed_terms), 0.5)


# Render the generated image given a tensorflow session and a variable image (x)
def render_img(session, x):
    shape = x.get_shape().as_list()
    toimage(np.reshape(session.run(x), shape[1:])).show()


# Save the generated image given a tensorflow session and variable image (x)
def save_img(session, x):
    shape = x.get_shape().as_list()
    toimage(np.reshape(session.run(x), shape[1:])).save(out_path)


with tf.Session() as sess:
    # Initialize and process photo image to be used for our content
    photo, image_shape = utils.load_image(input_path)
    image_shape = [1] + image_shape
    photo = photo.reshape(image_shape).astype(np.float32)

    # Initialize and process art image to be used for our style
    art, art_shape = utils.load_image(style_path)
    assert [1] + art_shape == image_shape, "Both the input and style images must be the same shape."
    art = art.reshape(image_shape).astype(np.float32)

    # Initialize the variable image that will become our final output as random noise
    noise = tf.Variable(tf.random_uniform(image_shape, minval=0, maxval=1))

    # VGG Networks Init
    with tf.name_scope('vgg_content'):
        content_model = vgg19.Vgg19()
        content_model.build(photo, image_shape[1:])

    with tf.name_scope('vgg_style'):
        style_model = vgg19.Vgg19()
        style_model.build(art, image_shape[1:])

    with tf.name_scope('vgg_x'):
        x_model = vgg19.Vgg19()
        x_model.build(noise, image_shape[1:])

    # Loss functions
    with tf.name_scope('loss'):
        # Content
        if content_weight is 0:
            content_loss = tf.constant(0.)
        else:
            content_loss = get_content_loss(x_model, content_model)
        weighted_content_loss = content_weight * content_loss

        # Style
        if style_weight is 0:
            style_loss = tf.constant(0.)
        else:
            style_loss = get_style_loss(x_model, style_model)
        weighted_style_loss = style_weight * style_loss

        # Norm regularization
        if norm_weight is 0:
            norm_loss = tf.constant(0.)
        else:
            norm_loss = get_norm(noise)
        weighted_norm_loss = norm_weight * norm_loss

        # Total variation denoising
        if tv_weight is 0:
            tv_loss = tf.constant(0.)
        else:
            tv_loss = get_total_variation(noise, image_shape)
        weighted_tv_loss = tv_weight * tv_loss

        # Total loss
        total_loss = tf.sqrt(weighted_content_loss + weighted_style_loss + weighted_norm_loss + weighted_tv_loss)

    # Update image
    with tf.name_scope('update_image'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads = optimizer.compute_gradients(total_loss, [noise])
        clipped_grads = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads]
        update_image = optimizer.apply_gradients(clipped_grads)

    # Train
    print("Initializing variables and beginning training..")
    sess.run(tf.initialize_all_variables())
    for i in range(epochs):
        if i % 250 == 0:
            print("Epoch %04d | Total Error is %s" % (i, sess.run(total_loss)))
            render_img(sess, noise)
        sess.run(update_image)

    # FIN
    print("Training complete. Rendering final image and closing TensorFlow session..")
    save_img(sess, noise.eval())
    sess.close()
