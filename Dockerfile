FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends gfortran \
     liblapack-dev \
     liblapack3 \
     libopenblas-base \
     libopenblas-dev \
     xorg \
     libx11-dev \
     libglu1-mesa-dev \
     libfreetype6-dev \
     software-properties-common \
     llvm-8-runtime \
     git

RUN pip3 install setuptools
RUN pip3 install --upgrade pip
RUN pip3 install Cython
RUN pip3 install numpy==1.23.2
RUN pip3 install numpy-quaternion==2020.11.2.17.0.49
RUN pip3 install argparse tqdm scipy scikit-learn 
RUN pip3 install opencv-python==4.4.0.46
RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"




#RUN ln /usr/local/cuda-8.0/lib64/stubs/libcuda.so /usr/local/cuda-8.0/lib64/stubs/libcuda.so.1
#RUN export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/stubs/libcuda.so.1:$LD_LIBRARY_PATH

