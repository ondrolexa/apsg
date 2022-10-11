#!/bin/bash

jupyter nbconvert --to notebook --inplace --execute 01_apsg_basics.ipynb
jupyter nbconvert --to notebook --inplace --execute 02_apsg_tensors.ipynb
jupyter nbconvert --to notebook --inplace --execute 03_apsg_stereonet.ipynb
jupyter nbconvert --to notebook --inplace --execute 04_apsg_fabricplots.ipynb
jupyter nbconvert --to notebook --inplace --execute 05_apsg_various.ipynb

