pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113  -f https://download.pytorch.org/whl/torch_stable.html -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
pip install -U openmim
mim install "mmcv-full<2.0.0"
pip install -r requirements.txt
cd projects/instance_segment_anything/ops
python setup.py build install
cd ../../..
cd mmdetection/
python3 setup.py build develop
cd ..