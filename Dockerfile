# syntax=docker/dockerfile:1

FROM ruhyadi/pytorch:v1.9.0-cu102

RUN apt-get update && apt-get install -y --no-install-recommends \
     libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 && \
     rm -rf /var/lib/apt/lists/*

WORKDIR /home

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Minimize image size 
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)  

ENV QT_X11_NO_MITSHM=1

CMD ["bash"]