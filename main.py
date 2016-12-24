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
epochs = 20000
learning_rate = 0.01
content_weight, style_weight, bias_weight = 0.001, 1.0, 1.25

# Process photo and art images for content and style learning
photo = utils.load_image(photo_path).reshape((1, 224, 224, 3)).astype(np.float32)
art = utils.load_image(art_path).reshape((1, 224, 224, 3)).astype(np.float32)
image_shape = [1, 224, 224, 3]


# Given an activated filter maps of any particular layer, return its respected gram matrix
def convert_to_gram(filter_maps):
    dimension = filter_maps.get_shape().as_list()
    reshaped_maps = tf.reshape(filter_maps, [dimension[1] * dimension[2], dimension[3]])

    # Compute the inner product to get the gram matrix
    if dimension[1] * dimension[2] > dimension[3]:
        return tf.matmul(reshaped_maps, reshaped_maps, transpose_a=True)
    else:
        return tf.matmul(reshaped_maps, reshaped_maps, transpose_b=True)


# Compute bias los
def get_bias_loss(x):
    with tf.name_scope('get_bias_loass'):
        height = image_shape[1]
        width = image_shape[2]
        a = tf.square(x[:, :height - 1, :width - 1, :] - x[:, 1:, :width - 1, :])
        b = tf.square(x[:, :height - 1, :width - 1, :] - x[:, :height - 1, 1:, :])
        c = tf.add(a, b)
        return tf.reduce_sum(tf.pow(c, 1.25))


# Given the photo activation filter maps of the photo and the generated image, return the content loss
def get_content_loss(x, c):
    with tf.name_scope('get_content_loss'):
        content_noise_out = x.conv4_2
        content_photo_out = c.conv4_2
        sum_of_squared_diffs = tf.reduce_sum(tf.square(tf.sub(content_noise_out, content_photo_out)))
        return tf.mul(0.5, sum_of_squared_diffs)


# Compute style loss
def get_style_loss(x_layers, art_layers):
    with tf.name_scope('get_style_loss'):
        style_layer_losses = [(get_style_loss_for_layer(xl, al)) for xl, al in zip(x_layers, art_layers)]

        # Weigh the layer losses and return their sum
        style_weights = tf.constant([0.2] * len(style_layer_losses), tf.float32)
        weighted_layer_losses = tf.mul(style_weights, tf.convert_to_tensor(style_layer_losses))
        return tf.reduce_sum(weighted_layer_losses)


# Compute style loss for a given layer
def get_style_loss_for_layer(xl, al):
    with tf.name_scope('get_style_loss_for_layer'):
        # Compute gram matrices using the activated filter maps of the art and generated images
        a_layer_gram = convert_to_gram(al)
        x_layer_gram = convert_to_gram(xl)

        # Compute the gram loss using the gram matrices
        style_layer_shape = xl.get_shape().as_list()
        style_layer_elements = ((style_layer_shape[1] * style_layer_shape[2]) ** 2) * (style_layer_shape[3] ** 2)
        style_fraction = 1.0 / (4.0 * style_layer_elements)
        gram_loss = tf.reduce_sum(tf.square(tf.sub(x_layer_gram, a_layer_gram)))

        # Compute the end layer loss as the weighted gram loss
        return tf.mul(style_fraction, gram_loss)


# Render the generated image given the session and image variable
def render_img(s, x):
    toimage(np.reshape(s.run(x), [224, 224, 3])).show()


with tf.Session() as sess:
    # Init
    noise = tf.Variable(tf.random_uniform(image_shape, minval=0, maxval=1))

    # VGG Networks Init
    content_model = vgg19.Vgg19()
    with tf.name_scope('vgg_content'):
        content_model.build(photo)

    style_model = vgg19.Vgg19()
    with tf.name_scope('vgg_style'):
        style_model.build(art)

    x_model = vgg19.Vgg19()
    with tf.name_scope('vgg_x'):
        x_model.build(noise)

    # Loss functions
    x_style_layers = [x_model.conv1_1, x_model.conv2_1, x_model.conv3_1, x_model.conv4_1, x_model.conv5_1]
    art_style_layers = [style_model.conv1_1, style_model.conv2_1, style_model.conv3_1, style_model.conv4_1, style_model.conv5_1]
    with tf.name_scope('loss'):
        weighted_bias_loss = tf.mul(bias_weight, get_bias_loss(noise))
        weighted_content_loss = tf.mul(content_weight, get_content_loss(x_model, content_model))
        weighted_style_loss = tf.mul(style_weight, get_style_loss(x_style_layers, art_style_layers))
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
    render_img(noise, noise.eval())
    sess.close()
