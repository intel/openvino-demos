# OpenVINO™ in AWS Marketplace

Currently, we offer [OpenVINO™ AMI in the AWS marketplace](https://aws.amazon.com/marketplace/pp/prodview-sa76mydxmlmwk). 

- Please see [Getting Started Guide to Launch EC2 with OpenVINO™](ami/Getting-Started-Guide-to-Launch-EC2-with-OpenVINO.pdf)
- See [Video Instructions to Launch OpenVINO AMI](https://youtu.be/ICdrUpWD26Q)
- See [Instructions to connect to OpenVINO AMI with RDP/VNC session](https://www.youtube.com/watch?v=K0eJshISQv4)

## Blog

[Click here](https://aws.amazon.com/blogs/industries/openvino-ami-now-available-on-aws-for-accelerating-oil-gas-exploration/) to read the blog for more details.

## Usage Instructions

1. [Launch the AMI.](https://aws.amazon.com/marketplace/pp/prodview-sa76mydxmlmwk)
   <br/>**Note:** Make sure the public IP address is enabled and is launched in a VPC with internet access.
   
1. Open Jupyter Notebook by navigating to port 8888,  the URL is `http://<ec2-instance-public-ip>:8888`
   <br/>**Note:** It might take a few minutes for the ec2 instance to boot up, so if the webpage doesn't load, please try again in a few minutes.
   
1. The Jupyter Notebook password is `<ec2-instance-id>`
   
1. To run sample notebooks, you can navigate to the `notebooks` folder.
<br/>Sample URL:  `http://<ec2-instance-public-ip>:8888/lab/tree/notebooks`


## Licensing

Please see [AWS AMI licensing agreements here.](ami/licensing)
