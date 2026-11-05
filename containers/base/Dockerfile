# MIT License
#
# Copyright (c) 2026 THALES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# A vendor base must be selected and pinned by digest by build_context.py.
# It must contain Python 3.12, pip, CMake, a C++17 compiler, and the target SDK.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Some vendor bases end with a non-root USER (openvino/ubuntu24_dev uses `openvino`), which cannot write
# the system Python environment. The offline install below needs root; operators must set a non-root user at run time.
USER root

ARG TARGET_PROFILE
LABEL org.passage.edge.profile=${TARGET_PROFILE}
ENV PASSAGE_PROFILE=${TARGET_PROFILE} \
    PASSAGE_NATIVE_LIBRARY=/opt/passage/lib/libpassage_inference.so \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PIP_NO_INDEX=1 \
    HOME=/tmp/passage-home XDG_CACHE_HOME=/tmp/passage-cache \
    NUMBA_CACHE_DIR=/tmp/passage-numba

WORKDIR /opt/passage
COPY wheels/ /opt/wheels/
COPY requirements.lock /opt/requirements.lock
COPY passage/ /opt/sources/passage/
COPY inference/ /opt/sources/inference/
COPY guard_runtime.py profiles.json entrypoint.py final_decoder_smoke.py /opt/passage/
COPY context-manifest.json /opt/passage/context-manifest.json

# No download or dependency resolution from the internet is allowed here.
# The Tiger OpenVINO developer base also includes openmodelzoo-modelapi, which
# requires NumPy <2 but is not used by PASSAGE. Remove this optional package
# before installing the locked NumPy 2 runtime; keep the OpenVINO SDK itself.
# Debian-managed setuptools/wheel (no RECORD file) cannot be uninstalled by pip, which aborts the install
# on Ubuntu-based vendor images. Install the two locked build tools beside them first (they shadow the
# distro copies, which stay untouched); the full locked install then finds them already satisfied.
RUN if [ "$TARGET_PROFILE" = "tiger" ]; then python3 -m pip uninstall -y openmodelzoo-modelapi; fi && \
    grep -E '^(setuptools|wheel)==' /opt/requirements.lock > /tmp/build-tools.lock && \
    python3 -m pip install --no-index --find-links=/opt/wheels --require-hashes --no-deps --ignore-installed -r /tmp/build-tools.lock && \
    rm /tmp/build-tools.lock && \
    python3 -m pip install --no-index --find-links=/opt/wheels --require-hashes -r /opt/requirements.lock && \
    python3 -m pip install --no-index --no-deps --no-build-isolation /opt/sources/passage /opt/sources/inference && \
    python3 -m pip check && \
    cmake -S /opt/sources/inference -B /tmp/passage-build -DPASSAGE_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/passage -DCMAKE_INSTALL_LIBDIR=lib && \
    cmake --build /tmp/passage-build --parallel 2 && cmake --install /tmp/passage-build && \
    python3 /opt/passage/guard_runtime.py --profile "$PASSAGE_PROFILE" && \
    python3 /opt/passage/final_decoder_smoke.py && \
    rm -rf /tmp/passage-build /opt/wheels

ENTRYPOINT ["python3", "/opt/passage/entrypoint.py"]
CMD ["passage-eval", "--help"]
