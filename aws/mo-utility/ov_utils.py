# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
#
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import tensorflow as tf
import tensorflow_hub as hub
import shutil
import sys
import os
from pathlib import Path
import docker
import zipfile
import boto3
from botocore.exceptions import ClientError

# Set log level to ERROR for tensorflow, PIL, IPKernelAPP
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def download_ov_image(docker_image, tag):
    docker_client = docker.from_env()
    try:
        # Check if docker image: openvino/ubuntu18_dev:tf2 exist locally
        _ = docker_client.images.get(docker_image + ":" + tag)
        print(f"{docker_image} Docker image found locally !")
    except:
        print(
            f"\nDownloading Docker image: {docker_image}:latest (about 5GB), This might take few minutes if this is the first time..."
        )
        try:
            docker_client.images.pull(docker_image, tag="latest")
            print(f"{docker_image}:latest Docker image downloaded successfully !")
        except Exception as err:
            sys.exit(f"{err} \n{docker_image}:latest Docker image download FAILED !")
        try:
            # Create docker image: openvino/ubuntu18_dev:tf2
            install_tf2_cmd = "pip install tensorflow -U"
            environment = []
            container_name = "ov-tf2"
            print(f"Installing latest tensorflow into {docker_image} ...")
            docker_output = docker_client.containers.run(
                name=container_name,
                image=docker_image + ":latest",
                remove=False,
                command=install_tf2_cmd,
                environment=environment,
                user="root",
            )

            # commit the container with latest TF installed
            ov_tf2_container = docker_client.containers.get(container_name)
            ov_tf2_container.commit(repository=docker_image, tag="tf2")
            ov_tf2_container.remove()
            print(f"Successfully created {docker_image}:tf2 !")
        except Exception as err:
            sys.exit(f"{err} \n{docker_image}:tf2 docker image creation FAILED !!")
            ov_tf2_container = docker_client.containers.get(container_name)
            ov_tf2_container.remove()


def download_keras_app_model(
    keras_app_model_name, keras_app_opts="(weights='imagenet')", output_dir="."
):
    keras_app_model_func = "".join(
        ["tf.keras.applications.", keras_app_model_name, keras_app_opts]
    )
    print(f"\nDownloading the model: {keras_app_model_name}...")
    keras_app_model = eval(keras_app_model_func)

    saved_model_dir, model_inp_shape = save_keras_model(keras_app_model, output_dir)
    return saved_model_dir, model_inp_shape


def download_tfhub_model(tfhub_model_url, input_shape, output_dir="."):
    print(f"\nDownloading the model: {tfhub_model_url} ...")
    tfhub_model = tf.keras.Sequential([hub.KerasLayer(tfhub_model_url)])
    tfhub_model.build(input_shape)
    saved_model_dir, model_inp_shape = save_keras_model(tfhub_model, output_dir)
    return saved_model_dir, model_inp_shape


def save_keras_model(keras_model, output_dir="."):
    model_inp_shape = keras_model.input.get_shape().as_list()
    # set batch size
    model_inp_shape[0] = 1
    model_inp_shape = str(model_inp_shape).replace(" ", "")

    saved_model_dir = "".join([output_dir])

    # Save the model
    if os.path.isdir(saved_model_dir):
        print(saved_model_dir, "exists already. Deleting the folder")
        shutil.rmtree(saved_model_dir)
    os.mkdir(saved_model_dir)

    print(f"\nSaving the model ...")
    keras_model.save(saved_model_dir)
    print(f"\nModel saved in SaveModel Format to {saved_model_dir} ")

    return saved_model_dir, model_inp_shape


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def create_ir_from_saved_model(
    saved_model_dir,
    model_inp_shape,
    mo_params,
):
    ir_model_name = mo_params["model_name"]
    ir_data_type = mo_params["data_type"]
    ir_model_dir = "IR_models/" + ir_data_type
    ir_output_path = "".join([saved_model_dir, "/", ir_model_dir])
    docker_saved_model_dir = "/mnt/"
    docker_ir_output_path = "".join([docker_saved_model_dir, "/", ir_model_dir])
    if model_inp_shape == "None":
        model_dir_list = [
            f
            for f in os.listdir(saved_model_dir)
            if os.path.isfile(os.path.join(saved_model_dir, f)) and f.endswith(".pb")
        ]
        config_list = [
            f
            for f in os.listdir(saved_model_dir)
            if os.path.isfile(os.path.join(saved_model_dir, f))
            and f.endswith(".config")
        ]
        docker_input_json = "".join(
            [
                "deployment_tools/model_optimizer/extensions/front/tf/",
                mo_params["input_json"],
            ]
        )

        if len(model_dir_list) > 1:
            print("There are more than one pb file")
        else:
            input_model = model_dir_list[0]

        docker_input_model = "".join([docker_saved_model_dir, input_model])

        if len(config_list) > 1:
            print("There are more than one config file")
        else:
            input_config = config_list[0]

        docker_input_config = "".join([docker_saved_model_dir, input_config])
        mo_cmd = f"mo_tf.py \
              --input_model {docker_input_model} \
              --output_dir {docker_ir_output_path}  \
              --tensorflow_object_detection_api_pipeline_config {docker_input_config} \
              --tensorflow_use_custom_operations_config {docker_input_json}"
    else:
        mo_cmd = f"mo_tf.py \
              --saved_model_dir {docker_saved_model_dir} \
              --input_shape {model_inp_shape} \
              --data_type {ir_data_type} \
              --output_dir {docker_ir_output_path}  \
              --model_name {ir_model_name}"

    if not os.path.exists(saved_model_dir):
        sys.exit(
            f"{saved_model_dir} missing. Make sure you have a TF model in SavedModel Format..."
        )

    if os.path.isdir(ir_output_path):
        print(f"{ir_output_path} exists already. Deleting the folder...")
        shutil.rmtree(ir_output_path)

    # Download OpenVINO docker image: "openvino/ubuntu18_dev:tf2".
    docker_client = docker.from_env()
    download_ov_image("openvino/ubuntu18_dev", "tf2")
    openvino_image = "openvino/ubuntu18_dev:tf2"

    print("\nStarting IR creation using OpenVINO model optimizer... ")
    print("\n--".join(mo_cmd.split("--")))
    print("\nPlease wait till the IR files are created... ")

    environment = [
        "CUDA_VISIBLE_DEVICES=-1",
        "PATH=/opt/intel/openvino/deployment_tools/model_optimizer:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH=/opt/intel/openvino/opencv/lib:/opt/intel/openvino/deployment_tools/ngraph/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/hddl_unite/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/hddl/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/gna/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/mkltiny_lnx/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib:/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64",
        "PYTHONPATH=/opt/intel/openvino/python/python3.6:/opt/intel/openvino/python/python3:/opt/intel/openvino/deployment_tools/tools/post_training_optimization_toolkit:/opt/intel/openvino/deployment_tools/open_model_zoo/tools/accuracy_checker:/opt/intel/openvino/deployment_tools/model_optimizer",
    ]

    # Run the OpenVINO model optimizer via Docker container
    docker_output = docker_client.containers.run(
        image=openvino_image,
        remove=True,
        command=mo_cmd,
        environment=environment,
        user="root",
        volumes={
            Path(saved_model_dir).absolute(): {
                "bind": docker_saved_model_dir,
                "mode": "rw",
            }
        },
    )

    print(docker_output.decode("utf-8"))

    if os.path.exists(ir_output_path):
        print(f"\nOpenVINO model saved in: {ir_output_path}")
        # Update permissions of the files created by docker.
        update_permissions_cmd = f"sudo chown $USER:$USER -R {saved_model_dir}"
        os.system(update_permissions_cmd)
    else:
        print("\nOpenVINO model creation FAILED ! ")


def upload_to_s3(output_dir, bucket_name):
    s3_client = boto3.client("s3")
    response = s3_client.list_buckets()
    bucket_list = []

    for bucket in response["Buckets"]:
        bucket_list.append(bucket["Name"])
    if bucket_name not in bucket_list:
        print(f"S3 bucket does not exist. Creating bucket {bucket_name}")
        region = boto3.Session().region_name
        if region == "us-east-1":
            s3_client1 = boto3.client("s3")
            s3_client1.create_bucket(Bucket=bucket_name)
        else:
            s3_client1 = boto3.client("s3", region_name=region)
            location = {"LocationConstraint": region}
            s3_client1.create_bucket(
                Bucket=bucket_name, CreateBucketConfiguration=location
            )
        print(f"Created S3 bucket {bucket_name}")

    zip_folder = output_dir.replace("./", "")
    file_name = f"{zip_folder}.zip"
    zipf = zipfile.ZipFile(file_name, "w", zipfile.ZIP_DEFLATED)
    zipdir(output_dir, zipf)
    zipf.close()
    response = s3_client.upload_file(file_name, bucket_name, file_name)
    print(f"Uploaded files to S3 bucket {bucket_name}")


def create_ir(create_ir_params):
    output_dir = create_ir_params.get("output_dir", ".")
    mo_params = create_ir_params.get("mo_params", ".")
    bucket_name = create_ir_params.get("bucket_name", ".")

    if create_ir_params.get("keras_app_model_name"):
        print(create_ir_params["keras_app_model_name"])
        keras_app_model_name = create_ir_params["keras_app_model_name"]
        keras_app_opts = create_ir_params.get("keras_app_opts", "(weights='imagenet')")
        saved_model_dir, model_inp_shape = download_keras_app_model(
            keras_app_model_name, keras_app_opts, output_dir
        )
    elif create_ir_params.get("objdet_model_url"):
        mo_params = create_ir_params.get("mo_params", ".")
        model_name = create_ir_params["mo_params"]["model_name"]
        url = create_ir_params["objdet_model_url"]
        url_command = f"wget '{url}'"
        tar_name = url.split("/")[-1]
        untar_command = f"tar -xvf {tar_name}"

        if os.path.isdir(output_dir):
            print(output_dir, "exists already. Deleting the folder")
            shutil.rmtree(output_dir)
        if os.path.exists(tar_name):
            print(tar_name, "exists already. Deleting it")
            os.remove(tar_name)

        exit_code = os.system(url_command)

        if exit_code != 0:
            print("Failed to Download model")
        else:
            print("Downloaded the model")
            untar_exit_code = os.system(untar_command)
            if untar_exit_code != 0:
                print("Failed to untar")
            else:
                print("Untarred the downloaded model")
                os.rename(model_name, output_dir)

        saved_model_dir = output_dir
        model_inp_shape = "None"
        print("Model: ", model_name)

    elif len(create_ir_params.get("mo_params", {}).get("input_shape")) > 2:
        input_shape = create_ir_params["mo_params"]["input_shape"]
        if create_ir_params.get("tfhub_model_url"):
            print(create_ir_params["tfhub_model_url"])
            saved_model_dir, model_inp_shape = download_tfhub_model(
                create_ir_params["tfhub_model_url"], input_shape, output_dir
            )
        elif create_ir_params.get("saved_model_dir"):
            print(create_ir_params.get("saved_model_dir"))
            saved_model_dir = create_ir_params["saved_model_dir"]
            model_inp_shape = create_ir_params["mo_params"]["input_shape"]
            # set batch size
            model_inp_shape[0] = 1
            model_inp_shape = str(model_inp_shape).replace(" ", "")
        else:
            sys.exit(
                "Either keras_app_model_name or tfhub_model_url or saved_model_dir should be given to create IR"
            )

    else:
        sys.exit(
            "Either keras_app_model_name or tfhub_model_url or objdet_model_url,input_shape or saved_model_dir,input_shape should be given to create IR"
        )

    create_ir_from_saved_model(saved_model_dir, model_inp_shape, mo_params)
    upload_to_s3(output_dir, bucket_name)