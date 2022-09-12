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

import shlex
import shutil
import sys
import os
import glob
import zipfile
from subprocess import run, PIPE
import boto3
import tensorflow as tf
import mo_tf
import tensorflow_hub as hub


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


def create_ir_from_saved_model(saved_model_dir, model_inp_shape, mo_params):
    ir_model_name = mo_params["model_name"]
    ir_data_type = mo_params["data_type"]
    ir_model_dir = "".join(["IR_models/", ir_data_type])
    ir_output_path = "".join([saved_model_dir, "/", ir_model_dir])
    mo_tf_file_path = mo_tf.__file__

    if model_inp_shape == "None":
        config_list = [
            f
            for f in os.listdir(saved_model_dir)
            if os.path.isfile(os.path.join(saved_model_dir, f))
            and f.endswith(".config")
        ]
        ir_input_json = "".join(
            [
                mo_tf_file_path.replace("mo_tf.py", ""),
                "mo/extensions/front/tf/",
                mo_params["input_json"],
            ]
        )

        if len(config_list) > 1:
            print("There are more than one config file")
        else:
            input_config = config_list[0]

        tf_obj_det_pipeline_config = "".join([saved_model_dir, "/", input_config])

        frozen_pb_path = "".join([saved_model_dir, "/", "frozen_inference_graph.pb"])
        saved_model_pb_path = "".join(
            [saved_model_dir, "/", "saved_model", "/", "saved_model.pb"]
        )

        if os.path.exists(frozen_pb_path):
            model_input = f"--input_model {frozen_pb_path}"
        elif os.path.exists(saved_model_pb_path):
            model_input = f"--saved_model_dir {saved_model_dir}/saved_model/"
        else:
            model_input = f"--saved_model_dir {saved_model_dir}"

        mo_cmd = f"python3 {mo_tf_file_path} \
              {model_input} \
              --output_dir {ir_output_path}  \
              --transformations_config  {ir_input_json} \
              --tensorflow_object_detection_api_pipeline_config {tf_obj_det_pipeline_config} \
              --reverse_input_channels "
    else:
        mo_cmd = f"mo \
              --saved_model_dir {saved_model_dir} \
              --input_shape {model_inp_shape} \
              --data_type {ir_data_type} \
              --output_dir {ir_output_path}  \
              --model_name {ir_model_name}  \
              {mo_params['mo_keras_arg']} "

    if not os.path.exists(saved_model_dir):
        sys.exit(
            f"{saved_model_dir} missing. Make sure you have a TF model in SavedModel Format..."
        )

    if os.path.isdir(ir_output_path):
        print(f"{ir_output_path} exists already. Deleting the folder...")
        shutil.rmtree(ir_output_path)

    print("\nStarting IR creation using OpenVINO model optimizer... ")
    print("\ \n--".join(mo_cmd.split("--")))
    print("\nPlease wait till the IR files are created... ")
    cmd = shlex.split(mo_cmd)

    try:
        return_args = run(cmd, stderr=PIPE, stdout=PIPE, text=True)
        print(return_args.stdout)
        print(return_args.stderr)
    except Exception as err:
        print(err)

    ov_ir_xml_path = glob.glob(f"{ir_output_path}/*.xml")
    if ov_ir_xml_path:
        print(f"\nOpenVINO model saved in: {ov_ir_xml_path}")
        # Update permissions of the files.
        update_permissions_cmd_str = f"sudo chown $USER:$USER -R {saved_model_dir}"
        update_permissions_cmd = shlex.split(update_permissions_cmd_str)
        run(update_permissions_cmd, stderr=PIPE, stdout=PIPE)
    else:
        err_msg = f"\n {ir_output_path} not created. OpenVINO IR creation FAILED ! "
        print(err_msg)
        raise Exception(err_msg)

    return return_args


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
    # for keras models, --disable_nhwc_to_nchw argument is needed for mo
    mo_params["mo_keras_arg"] = ""

    if create_ir_params.get("keras_app_model_name"):
        print(create_ir_params["keras_app_model_name"])
        mo_params["mo_keras_arg"] = "--disable_nhwc_to_nchw "
        keras_app_model_name = create_ir_params["keras_app_model_name"]
        keras_app_opts = create_ir_params.get("keras_app_opts", "(weights='imagenet')")
        saved_model_dir, model_inp_shape = download_keras_app_model(
            keras_app_model_name, keras_app_opts, output_dir
        )
    elif create_ir_params.get("objdet_model_url"):
        mo_params = create_ir_params.get("mo_params", ".")
        model_name = create_ir_params["mo_params"]["model_name"]
        url = create_ir_params["objdet_model_url"]
        url_command_str = f"wget '{url}'"
        url_command = shlex.split(url_command_str)
        tar_name = url.split("/")[-1]
        untar_command_str = f"tar -xvf {tar_name}"
        untar_command = shlex.split(untar_command_str)

        if os.path.isdir(output_dir):
            print(output_dir, "exists already. Deleting the folder")
            shutil.rmtree(output_dir)
        if os.path.exists(tar_name):
            print(tar_name, "exists already. Deleting it")
            os.remove(tar_name)

        download_cmd_out = run(url_command, stderr=PIPE, stdout=PIPE, text=True)

        if download_cmd_out.returncode != 0:
            print("Failed to Download model")
            print(download_cmd_out.stdout)
            print(download_cmd_out.stderr)
            sys.exit("Failed to Download model !")
        else:
            print("Downloaded the model")
            untar_cmd_out = run(untar_command, stderr=PIPE, stdout=PIPE, text=True)
            if untar_cmd_out.returncode != 0:
                print("Failed to untar")
                print(untar_cmd_out.stdout)
                print(untar_cmd_out.stderr)
                sys.exit("Failed to UNTAR the downloaded model !")
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
                "Either keras_app_model_name or tfhub_model_url or saved_model_dir \n"
                "should be given to create IR"
            )

    else:
        sys.exit(
            "Either keras_app_model_name or tfhub_model_url or objdet_model_url,\n"
            "input_shape or saved_model_dir,input_shape should be given to create IR"
        )

    create_ir_from_saved_model(saved_model_dir, model_inp_shape, mo_params)
    # upload_to_s3(output_dir, bucket_name)
