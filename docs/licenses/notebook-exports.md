# Notebook export licenses

The standalone pages under `docs/gen/notebooks/` use the `basic` HTML
template from Jupyter nbconvert 7.17.2. The template renders notebook cells
inside a PASSAGE page with a local stylesheet. The template's copyright
notices and license are reproduced with the exports. [nbconvert BSD 3-Clause license](nbconvert-7.17.2.txt)

The exported pages omit the JupyterLab theme bundle, including its embedded
icons, Material Design palette and shadow values. The routing objectives
notebook loads MathJax 2.7.7 from a content delivery network to render its
equations. The browser fetches that library separately. [MathJax 2.7.7 script](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML-full,Safe) ·
[MathJax Apache-2.0 license](https://github.com/mathjax/MathJax/blob/2.7.7/LICENSE)

The PASSAGE notebook sources and wrapper assets use the project's MIT
license. The dataset and its derived artifacts have their own attribution and
terms. [PASSAGE license](https://github.com/ThalesGroup/passage/blob/main/LICENSE) ·
[Dataset terms](../dataset/index.md#licence-and-attribution)
