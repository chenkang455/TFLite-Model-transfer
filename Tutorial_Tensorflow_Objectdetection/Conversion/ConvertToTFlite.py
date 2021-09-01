import tensorflow as tf

#Replace the path below to your own
saved_model_dir="Model/ExportModel/saved_model"
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('Model/TFlite/model.tflite', 'wb') as f:
  f.write(tflite_model)