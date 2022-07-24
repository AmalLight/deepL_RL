#!/bin/bash

wget https://bootstrap.pypa.io/get-pip.py

sudo apt install python3.9-distutils
sudo apt install python3.9-dev

python3.9 get-pip.py

pip3.9 cache purge
pip3.9 install lxml pexpect psutil requests simplejson dnspython
pip3.9 install mlagents
pip3.9 install tensorflow
pip3.9 install gym
pip3.9 install pandas torch matplotlib numpy scipy

# pip3 --default-timeout=10000 install ./python
