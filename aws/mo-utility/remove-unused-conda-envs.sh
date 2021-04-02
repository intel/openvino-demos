env_list=( "amazonei_mxnet_p27" "amazonei_mxnet_p36" "amazonei_pytorch_latest_p36" "amazonei_tensorflow2_p27" "amazonei_tensorflow2_p36" "amazonei_tensorflow_p27" "amazonei_tensorflow_p36" "chainer_p27" "chainer_p36" )

for env_name in "${env_list[@]}"
do
    conda env remove --name $env_name
done

echo "Successfully removed few unused conda envs. This will ensure we have enought space for docker images."