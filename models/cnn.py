import jax
import jax.numpy as jnp

def init_cnn_params(key, input_shape=(1, 28, 28), num_classes=10):
    """
    Initializes the parameters for a CNN with two convolutional layers
    and two fully connected layers.
    
    Parameters:
    - key (jax.random.PRNGKey): JAX random key for parameter initialization
    - input_shape (tuple): Shape of the input images (default is (1, 28, 28) for MNIST)
    - num_classes (int): Number of output classes (default is 10 for MNIST)

    Returns:
    - params (dict): Dictionary containing weights and biases for each layer
    """
    keys = jax.random.split(key, 4)  # Split key for each layer's parameters

    params = {
        'conv1': {
            'weights': jax.random.normal(keys[0], (8, 1, 3, 3)) * jnp.sqrt(2 / (3 * 3 * 1)),  # 1 input channel
            'biases': jnp.zeros(8)
        },
        
        'conv2': {
            'weights': jax.random.normal(keys[1], (16, 8, 3, 3)) * jnp.sqrt(2 / (3 * 3 * 8)),  # 8 input channels
            'biases': jnp.zeros(16)
        },
        'fc1': {
            'weights': jax.random.normal(keys[2], (784, 128)) * jnp.sqrt(2 / 784),
            'biases': jnp.zeros(128)
        },
        'fc2': {
            'weights': jax.random.normal(keys[3], (128, num_classes)) * jnp.sqrt(2 / 128),
            'biases': jnp.zeros(num_classes)
        }
    }
    return params

@jax.jit
def cnn_forward(params, x):
    """
    Forward pass for the CNN model.
    
    Parameters:
    - params (dict): Model parameters initialized by init_cnn_params
    - x (jnp.ndarray): Input data (batch of images)

    Returns:
    - jnp.ndarray: Output of the model after the forward pass
    """
    # First Convolutional Layer + ReLU + 2x2 Max Pooling
    x = jax.lax.conv_general_dilated(
        x, params['conv1']['weights'], window_strides=(1, 1), padding='SAME'
    )
    x = x + params['conv1']['biases'][None, :, None, None]
    x = jax.nn.relu(x)
    
    # 2x2 max pooling applied strictly to spatial dimensions
    x = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max,
        window_dimensions=(1, 1, 2, 2),  # Only pool over height and width
        window_strides=(1, 1, 2, 2),     # Stride only over height and width
        padding='VALID'
    )
    
    # Second Convolutional Layer + ReLU + 2x2 Max Pooling
    x = jax.lax.conv_general_dilated(
        x, params['conv2']['weights'], window_strides=(1, 1), padding='SAME'
    )
    x = x + params['conv2']['biases'][None, :, None, None]
    x = jax.nn.relu(x)
    
    # Apply 2x2 max pooling strictly to spatial dimensions
    x = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max,
        window_dimensions=(1, 1, 2, 2),  # Only pool over height and width
        window_strides=(1, 1, 2, 2),     # Stride only over height and width
        padding='VALID'
    )
    
    # Flatten for fully connected layers
    x = x.reshape(x.shape[0], -1)  # Flatten to (batch_size, -1) to fit fc1
    
    # First Fully Connected Layer + ReLU
    x = jnp.dot(x, params['fc1']['weights']) + params['fc1']['biases']
    x = jax.nn.relu(x)

    # Second Fully Connected Layer (Output)
    x = jnp.dot(x, params['fc2']['weights']) + params['fc2']['biases']
    return x  # Softmax can be applied in the loss function