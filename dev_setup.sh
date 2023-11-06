cd ..
git clone git@github.com:565353780/colmap-manage.git

cd colmap-manage
./dev_setup.sh

pip install numpy tqdm opencv-python Pillow imageio \
	pyyaml PyMCubes trimesh plyfile open3d tensorboardX \
	tensorboard
pip install jittor==1.3.6.15
