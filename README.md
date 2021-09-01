# Convert TF Object Detection Model  To TFLite 
There are two steps to convert TF Model to TFlite.  
## Step1: Export TFLite Inference Graph
This step generates a SavedModel which can be converted to TFLite using TFLite Converter.

Run this command and replace the pipeline_config_path,checkpoint_dir and your export directory.   
**Model structure:**
```
Conversion:
Model:
    checkpoint
        checkpoint
        ckpt-0.data-00000-of-00001
        ckpt-0.index
        ckpt-1.data-00000-of-00001
        ckpt-1.index
        ...
    pipeline.config
```
**Command:**
```python 
python Conversion/export_tflite_graph_tf2.py \
    --pipeline_config_path Model/pipeline.config\
    --trained_checkpoint_dir Model/checkpoints\
    --output_directory Model/ExportModel
```
Copy the command below and paste it on your command.
```
python Conversion/export_tflite_graph_tf2.py --pipeline_config_path Model/pipeline.config --trained_checkpoint_dir Model/checkpoints --output_directory Model/ExportModel
```

Waiting for a moment and there will be a folder named ExportModel under your Model folder.
```
Conversion:
Model:
    checkpoint(folder not file)
    ExportModel
        saved_model
            assets
            variables
            saved_model.pb
    pipeline.config
```

## Step2: Convert to TFLite
Using ConvertToTFlite.py and configure your saved model path and the export path or you can use the command line ,but the official strongly recommend the former mode.
```
python Conversion\ConvertToTFlite.py 
```
For the command line usage the official reference is very  detailed over there.
https://www.tensorflow.org/lite/convert

## Step3:Employ TFLite model
Having converted our model to model.tflite,we can employ it in some embedded devices like raspberry pi or android.(Remember to copy your TFlite model to the DetectOnCamera folder)   
Next,we are going to test our model on our windows10 system
```
python DetectOnCamera/detect.py
```
If all things going successfully,this code could detect whether the person wears the mask but the accuracy is relatively low owing to my little datasets.
![Mask](https://github.com/chenkang455/TFLite-Model-transfer/blob/main/images/Mask.png?raw=true)
![NoMask](https://github.com/chenkang455/TFLite-Model-transfer/blob/main/images/NoMask.png?raw=true)
##Common issues
This is the input format and the output format on tflite:
```
One input:
  image: a float32 tensor of shape[1, height, width, 3] containing the
  *normalized* input image.
Four Outputs:
  detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with box
  locations
  detection_classes: a float32 tensor of shape [1, num_boxes]
  with class indices
  detection_scores: a float32 tensor of shape [1, num_boxes]
  with class scores
  num_boxes: a float32 tensor of size 1 containing the number of detected boxes
```
Watch out the output format,different models may have different output formats which requires you to print(output_details) to watch its format and adjust the index parameters to adjust your model output format!!!
**UTF8 Code Error**
This means your file's root is not correct, you should examine it more carefully.

"# Convert-Tensorflow-to-Lite-OD" 
