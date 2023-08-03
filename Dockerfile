# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:2.0.4-gpu-py3

# Set the working directory as /work
WORKDIR /work

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Update the apt-get package lists
RUN apt-get update

# Install necessary packages
RUN apt-get install -y \
    wget \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# Upgrade pip
RUN pip3 install --upgrade pip

# Install IPython
RUN pip install IPython

RUN pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Add the requirements file to the container
ADD ./requirements.txt /work/requirements.txt

# Install the Python dependencies
RUN pip install -r /work/requirements.txt
RUN pip install --upgrade torch torchvision


# Clone the repository or pull the latest changes if it already exists
RUN if [ -d "Detecting_License_Plate" ]; then \
    cd Detecting_License_Plate && git pull; \
    else \
    git clone https://github.com/HonorJay/Detecting_License_Plate.git; \
    fi

# Add Detecting_License_Plate directory to PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/work/Detecting_License_Plate"
