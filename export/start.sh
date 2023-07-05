#!/bin/bash

set_py_env() {
    if [ $# -lt 4 ]
    then 
        echo "Not enough argument. Expected 4:"
        echo -e "argv[1] - name_proj\nargv[2] - install directory\nargv[3] - python version"
        exit
    fi
    proj=$1
    dir=$2
    python_verison=$3
    #
    #dow_pV="3.8.10"
    pV="python" $($python_version | cut -d"." -f 1-2)
    #pV="python3.8"
    #
    #sudo apt-get install python3
    if [ -d $dir ]
    then
        echo "$dir"
    else
        mkdir $dir
    fi
    cd $dir
    sudo wget https://www.python.org/ftp/python/$python_verison/Python-$python_verison.tgz
    sudo tar xzf Python-$python_verison.tgz
    cd Python-$python_verison
    sudo ./configure --enable-optimizations
    sudo make altinstall
    cd ..
    sudo rm -f Python-$python_verison.tgz
    #
    sudo apt-get install $pV-venv
    sudo apt-get install python3-pip
    #sudo pip install virtualenv
    mkdir $proj
    cd $proj/
    $pV -m venv $proj
    source $proj/bin/activate
    pip install --upgrade pip
    #pip install --upgrade python3
    pip install tensorflow-cpu,keras
    pip install pandas, numpy, astropy
    pip install sklearn, math
    pip install matplotlib, seaborn
    pip install os, sys, argparse, json, tempfile, requests, time, concurrent.futures, shutil, multiprocessing
    #pip install sklearn.cross_validation
    python -V
    pip freeze
    deactivate
}

set_py_env "$(pwd)/project_ml" "env" "3.11.4"
echo "done"
#chmod +x main.py
source project_ml/bin/activate
#python3.11 main.py -c config.json 