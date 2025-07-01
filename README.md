## How to update xtreme1's point clound detection model from CUDA-10.2 to CUDA-11.3 for AI annotating 

point cloud's model used for AI annotating in xtreme1 (here is xtreme1-v091-point-cloud-object-detection) uses CUDA-10.2 , and can't run on CUDA-11.3, if your computer's GPU is newer when you use xtreme1's AI annotating function,xtreme1 prompts `Model Run Error`, you need to update xtreme1's point cloud detect model to CUDA-11.3 or other higher verions to run AI annotating correctly. here are steps to do updating process.

### step 1: run the whole conatainer 
    $ docker compose --profile model up    
note that this command should be runned under xtreme1-v0.9.1 which conatain docker-compose.yml and deploy as the original project describled.if the container works well ,we'll see annotation web on http://localhost:8190 

### step 2: enter the point clound detection container 
    $ docker exec -it xtreme1-v091-point-cloud-object-detection-1 /bin/bash

to find which version of OS the image was used, use dpkg --list 
```
$root@d803d8bc1748:/app/pcdet_open# dpkg --list
||/ Name                                         Version                     Architecture                Description
+++-============================================-===========================-===========================-===================================
ii  adduser                                      3.116ubuntu1                all                         add and remove users and groups
ii  apt                                          1.6.14                      amd64                       commandline package manager
ii  apt-utils                                    1.6.14                      amd64                       package management related utility programs
ii  base-files                                   10.1ubuntu2.11              amd64                       Debian base system miscellaneous files
ii  base-passwd                                  3.5.44                      amd64                       Debian base system master password and group files
ii  bash                                         4.4.18-2ubuntu1.3           amd64                       GNU Bourne Again SHell
ii  binutils                                     2.30-21ubuntu1~18.04.7      amd64                       GNU assembler, linker and binary utilities
...
```
so we know ubuntu-18.04 was used to build this container
### step 3: check versions of Pytorch and CUDA which were used
```
$root@d803d8bc1748:/app/pcdet_open# pip list
Package             Version        Editable project location
------------------- -------------- -------------------------
cumm-cu102          0.2.9  
spconv-cu102        2.1.21
torch               1.10.1+cu102
torchvision         0.11.2+cu102
...
```
use pip to uninstall the above 4 CUDA 10.2 related packages
```
root@d803d8bc1748:/app/pcdet_open#  pip uninstall cumm-cu102
root@d803d8bc1748:/app/pcdet_open#  pip uninstall spconv-cu102
root@d803d8bc1748:/app/pcdet_open#  pip uninstall torch
root@d803d8bc1748:/app/pcdet_open#  pip uninstall torchvision
```
back to root folder /, and create upgrade folder for files used to update,  and  exit the container

```
root@d803d8bc1748:/app/pcdet_open# cd / 
root@d803d8bc1748:/app/pcdet_open# make upgrade 
root@d803d8bc1748:/app/pcdet_open# exit
```

### step 4: download ubuntu18.04's versions of CUDA11.3,CUDNN8.2,torchvision in host computer 
```
cuda_11.3.1_465.19.01_linux.run
torch-1.10.1+cu113-cp36-cp36m-linux_x86_64.whl
cudnn-11.3-linux-x64-v8.2.1.32.tgz 
torchvision-0.11.2+cu113-cp36-cp36m-linux_x86_64.whl
```
`cuda_11.3.1_465.19.01_linux.run` and `cudnn-11.3-linux-x64-v8.2.1.32.tgz`  can be downloaded from [here](https://developer.nvidia.com/cuda-toolkit-archive) and `torch-1.10.1+cu113-cp36-cp36m-linux_x86_64.whl`,`torchvision-0.11.2+cu113-cp36-cp36m-linux_x86_64.whl` can be downloaded [here](https://download.pytorch.org/whl/torch_stable.html)

cd to file folder which contained the downloaded files  
```
 $ ls -l
-rwxrwxrwx 1 hitbuyi hitbuyi 3158494112 5月  14  2021 cuda_11.3.1_465.19.01_linux.run
-rwxrwxrwx 1 hitbuyi hitbuyi 1879325034 6月  30 00:55 cudnn-11.3-linux-x64-v8.2.1.32.tgz
-rwxrwxrwx 1 hitbuyi hitbuyi 1821432505 6月  30 00:23 torch-1.10.1+cu113-cp36-cp36m-linux_x86_64.whl
-rwxrwxrwx 1 hitbuyi hitbuyi   24585457 6月  30 00:16 torchvision-0.11.2+cu113-cp36-cp36m-linux_x86_64.whl
```
copy files to container's upgrade folder

    $docker cp  ./  xtreme1-v091-point-cloud-object-detection-1:/upgrade/  


### step 5: enter the container again
    $ docker exec -it xtreme1-v091-point-cloud-object-detection-1 /bin/bash

 cd to upgrade,we the copied files here
 ```
 root@d803d8bc1748:/app/pcdet_open#  cd /upgrade
 root@d803d8bc1748:/upgrade# 
 root@d803d8bc1748:/upgrade#  ls -l 
 ```
 we found copied files
```
total 3992772
-rwxr-xr-x 1 root root  363235328 Jun 30 17:14 cuda_11.3.1_465.19.01_linux.run
-rwxrwxrwx 1 1000 1000 1879325034 Jun 29 16:55 cudnn-11.3-linux-x64-v8.2.1.32.tgz
-rwxrwxrwx 1 1000 1000 1821432505 Jun 29 16:23 torch-1.10.1+cu113-cp36-cp36m-linux_x86_64.whl
-rwxrwxrwx 1 1000 1000   24585457 Jun 29 16:16 torchvision-0.11.2+cu113-cp36-cp36m-linux_x86_64.whl
```
those *.whl files can be downloaded in official websites, I also put them on [here](www.baidu.com)
### step 6: install torch,torchvision,CUDA-11.3 and CUDNN-8.2 
#### 6.1 install torch,torchvision 
```
root@d803d8bc1748:/upgrade# pip install torch-1.10.1+cu113-cp36-cp36m-linux_x86_64.whl
root@d803d8bc1748:/upgrade# pip install torchvision-0.11.2+cu113-cp36-cp36m-linux_x86_64.whl
```
#### 6.2 instal CUDA-11.3 and CUDNN8.2
```
root@d803d8bc1748:/upgrade# chmod +x cuda_11.3.1_465.19.01_linux.run
root@d803d8bc1748:/upgrade#  sudo ./cuda_11.3.1_465.19.01_linux.run --silent --toolkit --override --installpath=/usr/local/cuda-11.3
```
note that CUDA-11.3 to be installed in /usr/local/cuda-11.3,wait for a while to finish CUDA-11.3 installment process 
install CUDNN

    root@d803d8bc1748:/upgrade# tar -xvf  cudnn-11.3-linux-x64-v8.2.1.32.tgz
a `cuda` folder was generated in current directory
```
root@d803d8bc1748:/upgrade#$  sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.3/include
root@d803d8bc1748:/upgrade#$  sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.3/lib64
root@d803d8bc1748:/upgrade#$  sudo chmod a+r /usr/local/cuda-11.3/include/cudnn*.h /usr/local/cuda-11.3/lib64/libcudnn*
```

use update-alternative to change CUDA from 10.2 to 11.3
```
root@d803d8bc1748:/upgrade#$ sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-10.2 113
root@d803d8bc1748:/upgrade#$ sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-11.3 120
root@d803d8bc1748:/upgrade#$ sudo update-alternatives --config cuda
There are 2 choices for the alternative cuda (providing /usr/local/cuda).

  Selection    Path                  Priority   Status
------------------------------------------------------------
  0            /usr/local/cuda-10.2   120       auto mode
  1            /usr/local/cuda-10.2   120       manual mode
* 2            /usr/local/cuda-11.3   113       manual mode

Press <enter> to keep the current choice[*], or type selection number: 
```
type 2 to select CUDA-11.3 
```
root@d803d8bc1748:/upgrade# nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Mon_May__3_19:15:13_PDT_2021
Cuda compilation tools, release 11.3, V11.3.109
Build cuda_11.3.r11.3/compiler.29920130_0
```
you will see CUDA version had been changed from 10.2 to 11.3

### step 7:install cumm-cu113,spconv-cu113
```
root@d803d8bc1748:/upgrade#$ pip install cumm-cu113
root@d803d8bc1748:/upgrade#$ pip install spconv-cu113
```
note that pip can download cumm-cu113 and spconv-cu113,though `ping` does not work. if your download speed is slow, I put spconv and other whl file used in step 4 [here](wwww.baidu.com)

after installments are finished, delete all the install files
```
root@d803d8bc1748:/upgrade#$ cd ..
root@d803d8bc1748:/upgrade#$ sudo rm  -rf ./upgrade
```

### step 8:restart the container
ctrl+c to stop the xtreme1-v091 started in step 1, and run:

    $ docker compose --profile model up

open http://localhost:8190, login into it ,open a dataset to annotate ,run AI annotating, no model error happens
![xtreme1 AI labeling](./images/AI%20Labelling.png) 