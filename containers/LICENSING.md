# Edge base recipes: licence review limits


## 1. Scope

These notes identify licence information observed during local preparation. They do not determine whether an image or any of its components may be redistributed. The public source files contain text recipes and locks, not model weights, wheels, Debian archives, OCI images or historical build contexts. The operator obtains the inputs separately, reviews the applicable terms and builds locally. PASSAGE does not distribute prebuilt images.

The standard PASSAGE install supports CPU evaluation. CUDA, TensorRT and
the other edge SDKs are separate operator-supplied inputs for explicitly
configured accelerator execution. PASSAGE distributes no prebuilt edge images.

## 2. First-party material

The PASSAGE source includes a THALES MIT licence. The separate inference source needed by the container builder is not currently public; access, an approved revision and its distribution rights remain to be resolved. A licence observed in a local checkout does not establish that the checkout or every file in it is authorized for public release.

## 3. Python wheels and SDKs

A local metadata inventory examined 80 ARM64 wheels and 79 Tiger wheels from historical contexts. Every examined wheel had a licence field, classifier or embedded licence file. That inventory was a metadata check; it did not establish compatibility, complete transitive notices or redistribution rights, and the historical contexts are not supplied here.

- The examined ARM64 `cuda-python` and `cuda-bindings` 13.2.0 wheels identified `LicenseRef-NVIDIA-SOFTWARE-LICENSE`. Review the full applicable NVIDIA terms for the intended use and distribution.
- The examined Tiger `openvino` 2026.4.1 wheel identified Apache-2.0. That declaration does not cover every component of an OpenVINO developer image or Intel GPU driver.
- The examined `hf_transfer` 0.1.9 wheels contained an embedded Apache 2.0 licence file, although the inspected metadata had no licence field or classifier.

Retain and review the licence texts and notices for the exact versions actually selected. Wheel metadata does not settle terms for separately supplied native libraries.

## 4. Images and OS packages

The Jetson SDK recipe describes a locally constructed image using Ubuntu and NVIDIA packages. The package lock is included; the package archives and full signed indexes are not. Independently verify vendor signing-key identities and review the licences and redistribution conditions for each selected package and component.

The Tiger developer image and its bundled components require their own licence and notice review, as do the target GPU drivers. A derived image or exported root filesystem is built and used locally; it is not distributed by PASSAGE.

## 5. Source and input review

Identify approved source revisions and owners, obtain the applicable permissions, and retain the required third-party licence texts and notices for each item distributed. Source rights, target execution, OCI identity and admission are separate questions; none follows merely from the recipes or package hashes.

For a locally built OCI image, review its base and every layer. A file copied in one
layer still forms part of the image if a later layer removes it.
