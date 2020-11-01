# Kaggle_GFootball
Files for Kaggle competition - Google Research Football with Manchester City F.C.


## Docker file
Docker file based on the tensorflow/tensorflow:latest-gpu-jupyter. At the moment of the creation the Tensorflow version was 2.3.1.

### Build Dockerfile
```
sudo docker build -t <name>:>version> .
```
  
### Start Docker container with GPU support
```
sudo docker run --gpus all -p 8888:8888 -v <path to the notebook>:/gfootball --name <name for new container> -it --rm <name>:<version>
```
  

