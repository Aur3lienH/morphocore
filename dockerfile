# Dockerfile to build a manylinux image with GCC 12 and CUDA 12.3
FROM sameli/manylinux_2_28_x86_64_cuda_12.3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN dnf install -y gcc-toolset-12 && \
    dnf clean all

ENV CC="/opt/rh/gcc-toolset-12/root/usr/bin/gcc"
ENV CXX="/opt/rh/gcc-toolset-12/root/usr/bin/g++"

RUN ln -sf /opt/rh/gcc-toolset-12/root/usr/bin/gcc /usr/bin/gcc && \
    ln -sf /opt/rh/gcc-toolset-12/root/usr/bin/g++ /usr/bin/g++ && \
    ln -sf /opt/rh/gcc-toolset-12/root/usr/bin/c++ /usr/bin/c++

RUN echo "source /opt/rh/gcc-toolset-12/enable" >> /root/.bashrc

