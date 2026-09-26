# llama.cpp fork de PrismML (necesario para Ternary Bonsai 2: el llama.cpp oficial
# carga el GGUF pero genera basura). Compilado para Jetson AGX Orin (sm_87, CUDA 13.2).
ARG CUDA_VERSION=13.2.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
ARG PRISM_TAG=prism-b10743-adfffbe
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${PRISM_TAG} https://github.com/PrismML-Eng/llama.cpp /src
WORKDIR /src
# libcuda.so real solo existe en runtime (lo inyecta nvidia-container-toolkit): enlazar contra el stub
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
       cmake -B build -G Ninja \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=87 \
        -DGGML_NATIVE=OFF \
        -DLLAMA_CURL=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs \
        -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs \
    && LIBRARY_PATH=/usr/local/cuda/lib64/stubs cmake --build build --target llama-server llama-bench llama-cli -j"$(nproc)"

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends libcurl4 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    # las libs cuda-compat de la imagen (595.45) rompen nvidia-cdi-hook en este host y sobran: el driver del host es más nuevo
    && rm -rf /usr/local/cuda/compat /usr/local/cuda/compat_orin
COPY --from=build /src/build/bin/ /app/
ENV LD_LIBRARY_PATH=/app
WORKDIR /app
ENTRYPOINT ["/app/llama-server"]
