FROM nvidia/cuda:10.1-runtime-ubuntu16.04
RUN  apt-get update -y && \
apt-get install -y python3-dev python3-pip  
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow==1.14
RUN pip3 install tflearn pandas numpy==1.16.4 sklearn matplotlib keras==2.3.1
RUN apt install vim -y
