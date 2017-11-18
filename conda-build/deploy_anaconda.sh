#!/bin/bash
# this script uses the CONDA_UPLOAD_TOKEN env var

rm -rf $HOME/miniconda/conda-bld

echo "Converting conda package..."
conda convert --platform osx-64 $HOME/miniconda/conda-bld/linux-64/apsg-*.tar.bz2 --output-dir $HOME/miniconda/conda-bld/
conda convert --platform linux-32 $HOME/miniconda/conda-bld/linux-64/apsg-*.tar.bz2 --output-dir $HOME/miniconda/conda-bld/
conda convert --platform win-32 $HOME/miniconda/conda-bld/linux-64/apsg-*.tar.bz2 --output-dir $HOME/miniconda/conda-bld/
conda convert --platform win-64 $HOME/miniconda/conda-bld/linux-64/apsg-*.tar.bz2 --output-dir $HOME/miniconda/conda-bld/
echo "Deploying to Anaconda.org..."
anaconda -t $CONDA_UPLOAD_TOKEN upload --force $HOME/miniconda/conda-bld/**/apsg-*.tar.bz2
echo "Successfully deployed to Anaconda.org."
exit 0
