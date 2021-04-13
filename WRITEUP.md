# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

Custom Layers are layers that are not in the list of supported layers.
Any layer not in that list is automatically classified as a custom layer by the Model Optimizer.

Custom Layers are registered as extensions to the model optimizer.

For Caffe, the second option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer. You’ll need Caffe on your system to do this option.

For TensorFlow, its second option is to actually replace the unsupported subgraph with a different subgraph. The final TensorFlow option is to actually offload the computation of the subgraph back to TensorFlow during inference.


## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

* Download the Model
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

* Untar It
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz 

* Convert it to IR
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

* Accuracy
The model is not accurate, in both cases. We can compare them, but I found they were similar. The person-detection-retail-0013 from Intel performed much better.

* The size of the model pre- and post-conversion is roughly similar ~65 Mb.

* The inference time is much lower when converted (0.080 ms after conversion)

## Assess Model Use Cases

Some of the potential use cases of the people counter app are counting the number of people at a strike, a presidential rally, or something similar.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

* Lighting: Too much light or not enough will affect the result (missing bounding boxes, duplicates, ...)
* Model accuracy: When deployed at the edge, we need a higher accuracy.
* Camera focal length: Focal length affects amount of information captured from image.If focal length is shorter it can cover wider area and captures less information from image.If focal length is longer then it covers shorter area and great amount of information from image.
* Image size: Bigger image size means better accuracy but bigger inference time (lower FPS). It's all about finding a tradeoff.

## Model Research

I tried Faster RCNN Resnet 50 and 101, there was several errors and I couldn't run them properly... after trying **a lot**.

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
