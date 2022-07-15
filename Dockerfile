FROM registry.deez.re/research/python-audio-gpu-11-2:latest

# libgoogle-perftools: cf. https://stackoverflow.com/questions/57180479/tensorflow-serving-oomkilled-or-evicted-pod-with-kubernetes
# poetry: wait until poetry is installed in python-audio img
RUN apt-get update && apt-get install -y ffmpeg \
    curl \
    libgoogle-perftools4 \
    && mkdir -p /var/cache \
    && mkdir -p /var/probes && touch /var/probes/ready \
    && pip install --upgrade --no-cache-dir poetry

# GSUTIL SDK
# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
# Installing the package
RUN mkdir -p /usr/local/gcloud \
    && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
    && /usr/local/gcloud/google-cloud-sdk/install.sh
# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# PYTHON Libraries
RUN pip install google-cloud-bigquery\
    google-cloud-storage\
    google-auth \
    ipython \
    hvac \
    pymysql  \
    dbutils \
    protobuf \
    setuptools \
    audioread \
    gin-config \
#    tensorflow==2.8.0 \
#    tensorflow-io \
    torchsummary \
    matplotlib \
    scipy \
    weightwatcher \
    tqdm \
    GPUtil \
    pandas \
    einops \
    speechbrain

RUN pip install torch==1.10.2+cu111 \
   torchvision==0.11.3+cu111 \ 
  torchaudio==0.10.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --extra-index-url https://artifacts.deez.re/repository/python-research/simple --trusted-host artifacts.deez.re deezer-audio[resampling] deezer-environment deezer-datasource

ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

USER deezer


