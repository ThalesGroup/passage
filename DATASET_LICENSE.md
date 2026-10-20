# PASSAGE Dataset Terms

## 1. Code and source data

The generator and evaluator code use the [MIT license](LICENSE).
The AW3D30-derived elevation data carry the applicable upstream terms, separately
from the code. This document summarizes those terms; it is not their full text
and does not grant additional rights over third-party material.

## 2. JAXA terms

The [official AW3D30 page, section 5](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm)
links to the [JAXA Terms of Use of Research Data](https://earth.jaxa.jp/en/data/policy/).
These pages were checked on 18 October 2026.

Under those terms, use, modification and redistribution are free of charge.
Credit JAXA and the product, including for publicly distributed derivatives;
retain applicable third-party credits. Commercial use requires prior notification
to JAXA. Notification of a research publication is requested where possible.
JAXA retains source-data copyright and disclaims liability. Its
[site policy](https://global.jaxa.jp/policy.html) also applies.

Credit line: **The original data used for this product have been supplied by JAXA's ALOS World 3D–30m (AW3D30).** Identify PASSAGE processing and the source revision
in the release metadata. Do not present modified terrain as an unmodified JAXA
product or the whole dataset as MIT-licensed or public domain.

AW3D30 fills voids with third-party elevation data. JAXA's [AW3D30 Version 4.1 Product Description](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/data/aw3d30v4.1_product_e_1.0.pdf) (checked on 19 October 2026; Table 1, mask file; Section 4, Related URLs) lists GSI Digital Topographic Map 5 m and 10 m mesh data (Geospatial Information Authority of Japan), SRTM-1 v3 (NASA/JPL), Viewfinder Panoramas DEM (Jonathan de Ferranti), ASTER GDEM v2 and v3 (NASA/METI), ArcticDEM v2 and v3 (NGA/NSF) and ArcticDEM v4, TanDEM-X 90m DEM (DLR), REMA v1.1 (PGC, University of Minnesota) and Copernicus DEM GLO-30 (ESA). For the Copernicus DEM it states: "AW3D30 is produced using Copernicus DEM GLO-30-F © DLR e.V. 2014-2018 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA, all rights reserved". Retain these credits in derived works.

## 3. Release preparation

Preserve this file, source credits and the [datasheet](DATASHEET.md) in each
mirror. Before publication, archive the applicable terms with the release and
check whether the commercial-use notification applies. No notification or upload
is performed by this documentation update.

Model weights need their own license and pretrained-model provenance; this file
does not determine their legal status automatically. PASSAGE is a research
benchmark. Its routes and evaluation do not certify operational safety.
