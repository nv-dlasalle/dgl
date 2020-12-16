from nvcr.io/nvidia/cuda:11.0-devel-ubuntu18.04 

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get -qy install python3 python3-pip software-properties-common wget

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - > /etc/apt/trusted.gpg.d/kitware.gpg
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get -qy install cmake 

RUN pip3 install nltk ogb rdflib torch pyinstrument

WORKDIR /workspace

RUN mkdir dgl

ADD cmake ./dgl/cmake
ADD CMakeLists.txt ./dgl/CMakeLists.txt
ADD include ./dgl/include
ADD python ./dgl/python
ADD src ./dgl/src
ADD tests ./dgl/tests
ADD third_party ./dgl/third_party
ADD tools ./dgl/tools

WORKDIR /workspace/dgl/

RUN mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON .. && make -j13
RUN cd python && python3 setup.py install

ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:/opt/conda/lib/python3.6/site-packages/torch/lib/"

ADD ./examples/pytorch/rgcn/entity_classify_mp.py ./examples/pytorch/rgcn/entity_classify_mp.py
ADD ./examples/pytorch/rgcn/model.py ./examples/pytorch/rgcn/model.py
ADD ./examples/pytorch/rgcn/utils.py ./examples/pytorch/rgcn/utils.py
ADD ./benchmark_rgcn.sh ./

ADD ./examples/pytorch/graphsage/load_graph.py ./examples/pytorch/graphsage/load_graph.py
ADD ./examples/pytorch/graphsage/train_sampling_multi_gpu.py ./examples/pytorch/graphsage/train_sampling_multi_gpu.py
ADD ./benchmark_graphsage.sh ./

