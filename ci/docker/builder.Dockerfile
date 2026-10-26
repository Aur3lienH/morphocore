# Builder image of the Linux wheels, for every torch version: manylinux_2_28 (gcc-toolset-14, which the
# recent torch headers need) with a CUDA 12 compiler. The wheel build jobs run in it directly, so the
# runner keeps it between jobs instead of downloading it and installing CUDA again in each one.
#
# Built and pushed by the builder-images job of .gitlab-ci.yml, from the ci/ directory:
#   docker build -f ci/docker/builder.Dockerfile --build-arg CUDA_VERSION=12.8 ci
FROM quay.io/pypa/manylinux_2_28_x86_64

ARG CUDA_VERSION
COPY install_cuda.sh /tmp/install_cuda.sh
RUN bash /tmp/install_cuda.sh ${CUDA_VERSION} && rm /tmp/install_cuda.sh

ENV CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:${PATH}
