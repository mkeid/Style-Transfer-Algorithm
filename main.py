#!/usr/bin/python
# Mohamed K. Eid (mohamedkeid@gmail.com)
import custom_vgg19 as vgg19
import tensorflow as tf
import numpy as np
import utils
from scipy.misc import toimage

# Model Hyper Params
photo_path = 'photo.jpg'
art_path = 'art.jpg'
content_layer = 'conv4_2'
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
content_weight, style_weight, bias_weight = 0.001, 1.0, 0.01
epochs = 20000
learning_rate = 0.01
total_variation_smoothing = 1.5


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


# Given the photo activation filter maps of the photo and the generated image, return the content loss
def get_content_loss(x, p):
    with tf.name_scope('get_content_loss'):
        # Get the activated VGG feature maps
        content_noise_out = getattr(x, content_layer)
        content_photo_out = getattr(p, content_layer)

        # Compute and return the loss
        sum_of_squared_diffs = tf.reduce_sum(tf.square(tf.sub(content_noise_out, content_photo_out)))
        return tf.mul(0.5, sum_of_squared_diffs)


# Compute style loss
def get_style_loss(x, a):
    with tf.name_scope('get_style_loss'):
        style_layer_losses = [(get_style_loss_for_layer(x, a, l)) for l in style_layers]
        style_weights = tf.constant([0.2] * len(style_layer_losses), tf.float32)
        weighted_layer_losses = tf.mul(style_weights, tf.convert_to_tensor(style_layer_losses))
        return tf.reduce_sum(weighted_layer_losses)


# Compute style loss for a given layer
def get_style_loss_for_layer(x, a, l):
    with tf.name_scope('get_style_loss_for_layer'):
        # Compute gram matrices using the activated filter maps of the art and generated images
        x_layer_maps = getattr(x, l)
        a_layer_maps = getattr(a, l)
        x_layer_gram = convert_to_gram(x_layer_maps)
        a_layer_gram = convert_to_gram(a_layer_maps)

        # Make sure the feature map dimensions are the same
        assert_equal_shapes = tf.assert_equal(x_layer_maps.get_shape(), a_layer_maps.get_shape())
        with tf.control_dependencies([assert_equal_shapes]):
            # Compute the gram loss using the gram matrices
            style_layer_shape = x_layer_maps.get_shape().as_list()
            style_layer_elements = ((style_layer_shape[1] * style_layer_shape[2]) ** 2) * (style_layer_shape[3] ** 2)
            style_fraction = 1.0 / (4.0 * style_layer_elements)
            gram_loss = tf.reduce_sum(tf.square(tf.sub(x_layer_gram, a_layer_gram)))

            # Compute the end layer loss as the weighted gram loss
            return tf.mul(style_fraction, gram_loss)


# Compute total variation regularization loss term
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
        return tf.reduce_sum(smoothed_summed_terms)


# Render the generated image given the session and image variable
def render_img(s, x):
    shape = x.get_shape().as_list()
    toimage(np.reshape(s.run(x), shape[1:])).show()


with tf.Session() as sess:
    # Initialize noise and process photo and art images for content and style learning
    image_shape = [1, 224, 224, 3]
    noise = tf.Variable(tf.random_uniform(image_shape, minval=0, maxval=1))
    photo = utils.load_image(photo_path).reshape(image_shape).astype(np.float32)
    art = utils.load_image(art_path).reshape(image_shape).astype(np.float32)

    # VGG Networks Init
    with tf.name_scope('vgg_photo'):
        photo_model = vgg19.Vgg19()
        photo_model.build(photo)

    with tf.name_scope('vgg_art'):
        art_model = vgg19.Vgg19()
        art_model.build(art)

    with tf.name_scope('vgg_x'):
        x_model = vgg19.Vgg19()
        x_model.build(noise)

    # Loss functions
    with tf.name_scope('loss'):
        weighted_bias_loss = tf.mul(bias_weight, get_total_variation(noise, image_shape))
        weighted_content_loss = tf.mul(content_weight, get_content_loss(x_model, photo_model))
        weighted_style_loss = tf.mul(style_weight, get_style_loss(x_model, art_model))
        total_loss = weighted_content_loss + weighted_style_loss + weighted_bias_loss

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
        if i % 50 == 0:
            print("Epoch %d | Total Error is %s" % (i, sess.run(total_loss)))
            render_img(sess, noise)
        sess.run(update_image)

    # FIN
    print("Training complete. Rendering final image and closing TensorFlow session..")
    render_img(sess, noise.eval())
    sess.close()
