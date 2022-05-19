env_list=( "amazonei_mxnet_p36" "amazonei_pytorch_latest_p37" "amazonei_tensorflow2_p36" "mxnet_p37" )

for env_name in "${env_list[@]}"
do
    conda env remove --name $env_name
done

echo "Successfully removed few unused conda envs. This will ensure we have enought space in the instance."