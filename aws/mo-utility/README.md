
# OpenVINO™ Model Optimization in AWS Sagemaker

This demonstrates a OpenVINO utility to convert Keras or TFHub models or TF object detection models in AWS Sagemaker notebook instance.

![ov-utils-arch.png](ov-utils-arch.png)

## Prerequisites

- AWS Sagemaker notebook instance. 
	- Recommended instance type: `ml.t3.xlarge`
	- Choose `notebook-al2-v1` as Platform identifier

![sagemaker-instance-selection.png](sagemaker-instance-selection.png)

## Instructions

1. Launch AWS Sagemaker Jupyer Notebook instance. See [Prerequisites](#Prerequisites).

2. Upload the [ov_utils.py](ov_utils.py) and [requirements.txt](requirements.txt)
3. Upload one of the jupyter notebook from this repo as per your requirement. See [Examples](#Examples).
4. Follow the instruction provided in Jupyter Notebook.

## Examples

- For Keras models, run [create_ir_for_keras.ipynb](create_ir_for_keras.ipynb). See [Keras-SupportedModelList.md](Keras-SupportedModelList.md)
- For TFHub models, run [create_ir_for_tfhub.ipynb](create_ir_for_tfhub.ipynb). See [TFHub-SupportedModelList.md](TFHub-SupportedModelList.md)
- For Object detection models, run [create_ir_for_obj_det.ipynb](create_ir_for_obj_det.ipynb). See [ObjDet-SupportedModelList.md](ObjDet-SupportedModelList.md)
