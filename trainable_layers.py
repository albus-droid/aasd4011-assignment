from tensorflow import keras
from tensorflow.keras import layers

def define_dense_model_with_hidden_layers(input_length, activation_func_array=['sigmoid', 'sigmoid'], hidden_layers_sizes=[50, 20], output_function='softmax', output_length=10):
    """Define a dense model with multiple hidden layers."""
    model = keras.Sequential()

    # Create the input layer
    model.add(layers.Dense(hidden_layers_sizes[0], input_shape=(input_length,), activation=activation_func_array[0]))

    # Create the hidden layers
    for size, activation in zip(hidden_layers_sizes[1:], activation_func_array[1:]):
        model.add(layers.Dense(size, activation=activation))

    # Create the output layer
    model.add(layers.Dense(output_length, activation=output_function))

    return model

def set_layers_to_trainable(model, trainable_layer_numbers):
    """Set specific layers of the model to trainable or non-trainable."""
    for i, layer in enumerate(model.layers):
        layer.trainable = i in trainable_layer_numbers
    return model

