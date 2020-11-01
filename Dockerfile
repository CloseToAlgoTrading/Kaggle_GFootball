FROM tensorflow/tensorflow:latest-gpu-jupyter

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get --no-install-recommends install -yq git cmake build-essential \
  libgl1-mesa-dev libsdl2-dev \
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
  libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
  libsdl-sge-dev python3-pip python3-dev apt-utils libsdl1.2-dev \
  libsdl-image1.2-dev libsdl-ttf2.0-dev libsdl-mixer1.2-dev libportmidi-dev \
  libsmpeg-dev python-numpy subversion ffmpeg \
  libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev wget \
  python3-setuptools python3-numpy python3-opengl \
  libtiff5-dev libx11-6 libx11-dev fluid-soundfont-gm timgm6mb-soundfont \
  xfonts-base xfonts-100dpi xfonts-75dpi xfonts-cyrillic fontconfig fonts-freefont-ttf 


SHELL ["/bin/bash", "-c"]
RUN python3 -m pip install --upgrade pip setuptools

RUN pip install pygame gfootball tensorflow tf-agents sklearn pandas kaggle_environments

WORKDIR "/gfootball"

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
