#!/bin/bash
sudo docker run \
 -it \
--name nomuraRandomForest \
--network=host \
--runtime nvidia -v /home/nomura/:/root/ nomurapi /bin/bash 
