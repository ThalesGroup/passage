# Edge base recipes: licence review limits

| Author | Date | Version |
| --- | --- | --- |
| Geoffrey Delhomme | 2026-11-04 | 0.2 — public source notes |

## 1. Scope

These notes identify licence information observed during local preparation. They do not determine whether an image or any of its components may be redistributed. The public source files contain text recipes and locks, not model weights, wheels, Debian archives, OCI images or historical build contexts. The operator must review the actual inputs and obtain any missing rights and notices before distribution.

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

The Tiger developer image and its bundled components require their own licence and notice review, as do the target GPU drivers. A derived image or exported root filesystem containing candidate weights must remain private until the applicable model, weight, architecture and SDK rights are resolved.

## 5. Before distribution

Identify approved source revisions and owners, obtain the applicable permissions, and retain the required third-party licence texts and notices for each item distributed. Source rights, target execution, OCI identity and admission are separate questions; none follows merely from the recipes or package hashes.
