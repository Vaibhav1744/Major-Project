from tensorflow.keras.models import load_model

# Load your model
model = load_model("Emotions_Model.h5")

# Print model summary
model.summary()

# Print number of classes
print("Output shape:", model.output_shape)
print("Number of classes:", model.output_shape[-1])
