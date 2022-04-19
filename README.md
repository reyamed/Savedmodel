# Savedmodel
## Introduction 
- this is an SSD-mobilenet V2 fine tuned object detection model trained on a custom dataset (13 classes) tensorflow 2.5.
## steps of training
1- create a new virtual environnement 
```sh
      Conda create -n nameproject python=3.6
```
2. install object detection api with tensorflow: 
```sh
     git clone https://github.com/tensorflow/models
```
3. proceeds to training 

## Deployment on OpenCV
### important information
1. the transfer learning was done on the pretrained model ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
2. the model was trained using model_main_tf2.py
3. the model was exported using exporter_main_v2.py
4. Since OpenCV doesn't support Saved_model format and my model was trained on tensorflow 2.5 I couldn't export the model graph directly because tensorflow gave on the graph execution on tensorflow 2. So I took an alternate route.
5. I freezed the model using the above python script Freeze_model.py
6. I optimized the model using the below command with tensorflow 1.15
```sh
     python -m tensorflow.python.tools.optimize_for_inference --input ./frozen_graph_final.pb --output ./optmized_graph_final.pb --frozen_graph=True --input_names="x" --output_names="Identity,Identity_1,Identity_2,Identity_3,Identity_4,Identity_5,Identity_6,Identity_7" 

```
7. export the .pbtxt using the above script export-to-pbtxt.py
