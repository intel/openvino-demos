# Readme for Installing OpenVINO in IBM CP4D v3.5.0

In this Readme, we show two methods to install OpenVINO in IBM CP4D (v3.5.0) Watson Studio Jupyter Environment:

1. By using conda and pip directly in a notebook
1. By building a customized image

For detailed instructions see: [Customizing environment definitions (Watson Studio)](https://www.ibm.com/support/producthub/icpdata/docs/content/SSQNUZ_latest/wsj/analyze-data/cust-env-parent.html)

## 1. By using conda and pip directly in a notebook

Run the following statements in your IBM CP4D Jupyter notebook. Use **Python 3.6 environment.**

OpenVINO currently supports Python 3.6, 3.8 on Red Hat Enterprise Linux 8, 64-bit.

**See [this Sample Notebook](ov-install-ibm-cp4d-jupyter.ipynb) for detailed instructions.**

```bash
# Install this specific version of OpenCV to prevent libGl errors
!pip uninstall -y opencv-python
!pip install -U opencv-python-headless==4.2.0.32 --user

# Install OpenVINO
!pip install --ignore-installed PyYAML openvino-dev

# Verify Installation
!pip show openvino-dev

# Restart Kernel, just to make sure all the paths are updated.
```


## 2. Creating and registering a custom image

**For detailed instructions see:** [Customizing environments > Building custom images](https://www.ibm.com/support/producthub/icpdata/docs/content/SSQNUZ_latest/wsj/analyze-data/build-cust-images.html)

**Required role:** You must be a **Cloud Pak for Data cluster administrator** to create and register a custom image.

**Required software:**

1. Install OC: [https://github.com/openshift/origin/releases](https://github.com/openshift/origin/releases)
2. Docker
3. Terminal to run the following commands.

### Steps to create and register a custom image:

1. Login into OpenShift cluster and get OC login credentials.

1. Login into cluster

    ```bash
    oc login \
    --token=sha256~abcr1pABCDEFGHIJabcwdA9990ABCt0134abcdefghI \
    --server=https://c1.us.server.cloud.jade.com:30999
    ```

1. Prepare to build a new image by:
    - **Get the registry URL** to use for Docker commands and in scripts. The Watson Studio runtime images are stored in a Docker image registry. In Cloud Pak for Data, you can only use an external registry outside of the Cloud Pak for Data OpenShift cluster.

    - To use that registry, you need the URL to the external registry that was used during the installation of Cloud Pak for Data. You use the same URL for all commands and in all scripts that you run.

    ```bash
    oc -n openshift-image-registry get route
    ```

    - Output will look like:

    ```bash
    NAME            HOST/PORT                                                                                             PATH      SERVICES         PORT      TERMINATION   WILDCARD
    default-route   default-route-openshift-image-registry.tester-cloud-99-0000.us.containers.appdomain.cloud             image-registry   <all>     reencrypt     None
    ```

    - **Set the following variables** for ease of use

    ```bash
    openshift_image_registry_route='default-route-openshift-image-registry.tester-cloud-99-0000.us.containers.appdomain.cloud'
    cloudPakforData_URL='https://jade-cpd.tester-cloud-99-0000.us.containers.appdomain.cloud'
    runtime_config_server_json='jupyter-py36-server.json'
    uname='admin'
    pwd='SuperSecret'
    ```

    - **Get Access Token**

    ```bash
    curl ${cloudPakforData_URL}/v1/preauth/validateAuth -u ${uname}:${pwd}

    myToken=`curl -k ${cloudPakforData_URL}/v1/preauth/validateAuth -u ${uname}:${pwd} | sed -n -e 's/^.*accessToken":"//p' | cut -d'"' -f1`
    ```

    - **Download the configuration file** for the runtime image that you want to customize. See [Downloading the configuration file](https://www.ibm.com/support/producthub/icpdata/docs/content/SSQNUZ_latest/wsj/analyze-data/download-runtime-def.html), for list of JSON configuration files. In this README we chose  `jupyter-py36-server.json`

    ```bash
    curl -X GET \
    "${cloudPakforData_URL}/zen-data/v1/volumes/files/%2F_global_%2Fconfig%2F.runtime-definitions%2Fibm%2F${runtime_config_server_json}" \
    --header "Authorization: Bearer ${myToken}" -k \
    >> ${runtime_config_server_json}
    ```

    - **Login to Docker** - Authenticate to perform Docker actions

    ```bash
    docker login -u kubeadmin -p $(oc whoami -t) ${openshift_image_registry_route}
    ```

    - **Download the image** in the configuration. See [Downloading the image.](https://www.ibm.com/support/producthub/icpdata/docs/content/SSQNUZ_latest/wsj/analyze-data/download-base-image.html)

    ```bash
    docker pull ${openshift_image_registry_route}/zen/wslocal-runtime-py36main:3.5.2011.1800-amd64
    ```

1. Adding customizations and building a new image. See [Creating a custom image.](https://www.ibm.com/support/producthub/icpdata/docs/content/SSQNUZ_latest/wsj/analyze-data/create-customized-image.html)

    ```bash
    # docker build \
    -t <new-image-name>:<new-image-tag> \
    --build-arg base_image_tag= <your_image_registry_location>/wslocal-x86-runtime-python36:master-273 \
    -f <path_to_dockerfile> .
    ```

    - Using the base image, we add OpenVINO installation commands. See [Dockerfile.cp4d.openvino](Dockerfile.cp4d.openvino). Build a new image with OpenVINO.

    ```bash
    docker build \
    -t wslocal-runtime-py36main-ov:3.5.2011.1800-amd64-ov \
    --build-arg base_image_tag=${openshift_image_registry_route}/zen/wslocal-runtime-py36main:3.5.2011.1800-amd64 \
    -f Dockerfile.cp4d.openvino .
    ```

1. Test docker image locally

    ```bash
    docker run -it --rm wslocal-runtime-py36main-ov:3.5.2011.1800-amd64-ov bash
    # Now you will be inside a docker container. Run the following to verify if OpenVINO is installed.
    conda env list
    conda init bash
    source ~/.bashrc
    conda activate Python-3.6-WMLCE
    pip show openvino-dev
    # Contol+D to exit the docker container
    ```

1. Pushing the image to the container server. See [Registering the custom image.](https://www.ibm.com/support/producthub/icpdata/docs/content/SSQNUZ_latest/wsj/analyze-data/register-image.html)

    ```bash
    # docker tag <new-image-name> <target_registry>/<new-image-name>:<new-image-tag>
    # docker push <target_registry>/<new-image-name>:<new-image-tag>
    ```

    ```bash
    docker tag \
    wslocal-runtime-py36main-ov:3.5.2011.1800-amd64-ov \
    ${openshift_image_registry_route}/zen/wslocal-runtime-py36main-ov:3.5.2011.1800-amd64-ov

    docker push \
    ${openshift_image_registry_route}/zen/wslocal-runtime-py36main-ov:3.5.2011.1800-amd64-ov
    ```

1. Changing and uploading the configuration file. See [Uploading the change configuration.](https://www.ibm.com/support/producthub/icpdata/docs/content/SSQNUZ_latest/wsj/analyze-data/upload-runtime-def.html)

    ```bash
    cp jupyter-py36-server.json jupyter-py36-ov2021-server.json
    ```

    **Edit the** `jupyter-py36-ov2021-server.json` file with the following. See [jupyter-py36-ov2021-server.json](jupyter-py36-ov2021-server.json)

    ```bash
    "image": "image-registry.openshift-image-registry.svc:5000/zen/wslocal-runtime-py36main-ov:3.5.2011.1800-amd64-ov",
    ```

    **Upload the configuration file.** Get a new token if needed. See `Get Access Token` in Step 3 above.

    ```bash
    curl -X PUT \
   "${cloudPakforData_URL}/zen-data/v1/volumes/files/%2F_global_%2Fconfig%2F.runtime-definitions%2Fibm" \
   -H "Authorization: Bearer ${myToken}" \
   -H "content-type: multipart/form-data" \
   -F upFile=@jupyter-py36-ov2021-server.json -k
    ```

1. After this, you should be able to see this newly created environment defintion when you create a new environment.
