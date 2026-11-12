# Inference assets

- Executable: `bin/yolox_demo`; Linux ARM64 command-line/JSON demo variant from the Amlogic model playground.
- Model: `models/yolox_s_w8a8.adla`; A311Y3 ADLA inference asset from the existing Amlogic device distribution.
- Source project: https://github.com/Amlogic-NN/amlnn-model-playground . The demo source files retain Amlogic copyright and Apache-2.0 notices; a copy of that license is in `licenses/Apache-2.0.txt`.
- The executables incorporate OpenCV code. Available third-party notices from the Amlogic OpenCV distribution are retained under `licenses/opencv/`; OpenCV Apache-2.0 terms are included in `licenses/Apache-2.0.txt`.
- A compatible device-provided `libnnsdk.so` and ADLA driver are external runtime dependencies and are not redistributed here.

The model assets keep their original distribution terms; this file does not grant an additional model or SDK license.
