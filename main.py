import argparse
import torch
import torchvision
from torchvision import transforms
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import os

from data.mnist_loader import load_mnist_data
from models.mlp import init_mlp_params, mlp_forward
from models.cnn import init_cnn_params, cnn_forward

# Argument parsing for model type and number of epochs
parser = argparse.ArgumentParser(description="Train an MLP or CNN on MNIST dataset using JAX.")
parser.add_argument('--model_type', type=str, choices=['MLP', 'CNN'], default='MLP',
                    help="Specify the model type: 'MLP' for Multi-Layer Perceptron or 'CNN' for Convolutional Neural Network.")
parser.add_argument('--num_epochs', type=int, default=100, 
                    help="Specify the number of training epochs.")

args = parser.parse_args()

MODEL_TYPE = args.model_type  # Use the specified model type
NUM_EPOCHS = args.num_epochs  # Use the specified number of epochs

# Set parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
PRINT_EVERY = 1
INPUT_SHAPE = (1, 28, 28)  # Shape of MNIST images

print("Loading MNIST data...")
# Load MNIST data
train_images, train_labels, test_images, test_labels = load_mnist_data(
    batch_size=BATCH_SIZE
)
print("Data loaded successfully.\n")

# Reshape the images for CNN if needed
if MODEL_TYPE == "CNN":
    train_images = train_images.reshape(-1, 1, 28, 28)
    test_images = test_images.reshape(-1, 1, 28, 28)

# Move data to the device (GPU or CPU) for efficient computation
train_images = jax.device_put(train_images)
train_labels = jax.device_put(train_labels)
test_images = jax.device_put(test_images)
test_labels = jax.device_put(test_labels)

# Initialize parameters based on the chosen model
print(f"Initializing {MODEL_TYPE} model parameters...")
key = jax.random.key(42)
if MODEL_TYPE == "MLP":
    layer_widths = [784, 256, 128, 10]
    params = init_mlp_params(layer_widths, key)
    forward = mlp_forward
elif MODEL_TYPE == "CNN":
    params = init_cnn_params(key, input_shape=INPUT_SHAPE, num_classes=10)
    forward = cnn_forward
else:
    raise ValueError("MODEL_TYPE must be 'MLP' or 'CNN'")
print(f"{MODEL_TYPE} model initialized.\n")


# Define the cross-entropy loss function
@jax.jit
def loss_fn(params, x, y):
    predictions = jax.nn.softmax(forward(params, x), axis=-1)
    return -jnp.mean(jnp.sum(y * jnp.log(predictions + 1e-10), axis=1))


# Update function using JAX's jit and grad
@jax.jit
def update(params, x, y, lr=LEARNING_RATE):
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)


# Define accuracy calculation
def compute_accuracy(params, x, y):
    logits = forward(params, x)
    predicted_classes = jnp.argmax(logits, axis=1)
    true_classes = jnp.argmax(y, axis=1)
    return jnp.mean(predicted_classes == true_classes)


print("Starting training...\n")
# Training loop
loss_history = []
accuracy_history = []

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()  # Record start time for the epoch

    # Training step
    for i in range(0, len(train_images), BATCH_SIZE):
        x_batch = train_images[i : i + BATCH_SIZE]
        y_batch = train_labels[i : i + BATCH_SIZE]

        # Ensure last batch size compatibility for CNN if smaller than BATCH_SIZE
        if MODEL_TYPE == "CNN" and x_batch.shape[0] != BATCH_SIZE:
            x_batch = x_batch.reshape(-1, 1, 28, 28)

        params = update(params, x_batch, y_batch)

    # Calculate epoch duration in seconds
    epoch_duration = time.time() - epoch_start_time

    # Evaluate loss on the full training set for monitoring
    current_loss = loss_fn(params, train_images, train_labels)
    loss_history.append(current_loss)

    # Evaluate accuracy on the test set
    test_accuracy = compute_accuracy(params, test_images, test_labels)
    accuracy_history.append(test_accuracy)

    # Print progress every PRINT_EVERY epochs
    if (epoch + 1) % PRINT_EVERY == 0:
        print(
            f"Epoch {epoch + 1}, Loss: {current_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%, Time per epoch: {epoch_duration:.2f} sec/epoch"
        )

print(f"Training complete. Final test accuracy: {test_accuracy * 100:.2f}%")


# Create results directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Generate the filename using model type and number of epochs
filename = f"plots/{MODEL_TYPE}_epochs{NUM_EPOCHS:03}.png"

# Plotting the training loss and test accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{MODEL_TYPE} Training Loss")

plt.subplot(1, 2, 2)
plt.plot(accuracy_history)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"{MODEL_TYPE} Test Accuracy")

# Save the plot with model name and number of epochs in the filename
plt.savefig(filename)
print(f"\nPlot showing Training Loss and Test Accuracy saved to {filename}")

# plt.show()
