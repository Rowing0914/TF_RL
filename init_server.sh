# apt installs
add-apt-repository ppa:jonathonf/python-3.6
apt-get update
apt-get install -y git python3.6 vim git-core python-opencv python3-pip

# set alias for pip3.6
echo "\n" >> ~/.bashrc
echo "alias pip3.6='python3.6 -m pip'" >> ~/.bashrc
export LC_ALL=C
source ~/.bashrc
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev

pip3.6 install --index-url https://test.pypi.org/simple/ --no-deps TF_RL
pip3.6 install gym tensorflow-gpu==1.14.0 numpy opencv-python matplotlib ipykernel pandas ray tensorflow_probability
pip3.6 install --upgrade gym[atari]
