#### FOR NOW I WILL ADD MODALAI'S SCRIPT ... CHANGE LATER TO THE SCRIPT THAT WORKS #### 

# IF ON VOXL 1, MAKE SURE TF VERSION IS <= 2.2.3
# i.e., pip install tensorflow==2.2.3
import tensorflow as tf

# if you have a saved model and not a frozen graph, see: 
# tf.compat.v1.lite.TFLiteConverter.from_saved_model()

# INPUT_ARRAYS, INPUT_SHAPES, and OUTPUT_ARRAYS may vary per model
# please check these by opening up your frozen graph/saved model in a tool like netron
converter =  tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file = '/path/to/.pb/file/tflite_graph.pb', 
  input_arrays = ['normalized_input_image_tensor'],
  input_shapes={'normalized_input_image_tensor': [1,300,300,3]},
  output_arrays = ['TFLite_Detection_PostProcess', 'TFLite_Detection_PostProcess:1', 'TFLite_Detection_PostProcess:2', 'TFLite_Detection_PostProcess:3'] 
)

# IMPORTANT: FLAGS MUST BE SET BELOW #
converter.use_experimental_new_converter = True
converter.allow_custom_ops = True
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
with tf.io.gfile.GFile('model_converted.tflite', 'wb') as f:
  f.write(tflite_model)
