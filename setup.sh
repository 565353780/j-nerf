cd ..
git clone https://github.com/565353780/colmap-manage.git

cd colmap-manage
./setup.sh

pip install numpy tqdm opencv-python Pillow imageio \
	pyyaml PyMCubes trimesh plyfile open3d
pip install jittor==1.3.6.15
