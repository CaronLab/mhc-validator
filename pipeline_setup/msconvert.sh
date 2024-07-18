#!/bin/bash
sudo docker run -it --rm -v /home/USER/FOLDER/TO_YOUR_RAW_FILES:/data proteowizard/pwiz-skyline-i-agree-to-the-vendor-licenses wine msconvert --filter "peakPicking true 1-2" /data/*.raw --mgf
