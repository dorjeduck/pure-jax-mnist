import numpy as np
import jax
import jax.numpy as jnp

def init_mlp_params(layer_widths, key):
    """
    Initializes the parameters for a fully connected (dense) MLP model.
    
    Parameters:
    - layer_widths (list of int): Specifies the width of each layer, e.g., [784, 256, 128, 10]
    - key (jax.random.PRNGKey): JAX random key for parameter initialization

    Returns:
    - params (list of dict): List of dictionaries containing weights and biases for each layer
    """
    params = []
    keys = jax.random.split(key, len(layer_widths) - 1)  # Split key for each layer's parameters

    for n_in, n_out, k in zip(layer_widths[:-1], layer_widths[1:], keys):
        layer_params = {
            'weights': jax.random.normal(k, (n_in, n_out)) * np.sqrt(2.0 / n_in),
            'biases': jnp.ones(shape=(n_out,))
        }
        params.append(layer_params)
    
    return params

def mlp_forward(params, x):
    """
    Performs the forward pass through the MLP.
    
    Parameters:
    - params (list of dict): Model parameters initialized by init_mlp_params
    - x (jnp.ndarray): Input data

    Returns:
    - jnp.ndarray: Output of the model after the forward pass
    """
    *hidden_layers, last_layer = params

    # Forward pass through hidden layers with ReLU activation
    for layer in hidden_layers:
        x = jax.nn.relu(jnp.dot(x, layer['weights']) + layer['biases'])

    # Final layer with no activation (softmax typically applied after this in loss function)
    x = jnp.dot(x, last_layer['weights']) + last_layer['biases']
    return x