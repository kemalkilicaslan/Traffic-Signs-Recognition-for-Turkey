# Traffic Signs Recognition for Turkey

A list of 17 frequently encountered traffic identification signs in urban areas in Turkey

- bumpy road
- bus stop
- give way
- green light
- no left turn
- no right turn
- no u turn
- parking
- parking is forbidden
- pedestrian crossing
- red light
- road with no entrance
- school crossing
- stop
- stopping and parking is forbidden
- vehicle is towed
- yellow light



It is an individual project that I have developed to be used in real time for the control and automation of autonomous vehicles with the model I trained on 668 images using the [roboflow](https://universe.roboflow.com/kemalkilicaslan-zdxtt/trafficsignsrecognitionforturkey-enyxp) platform and video processing with [ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8x.

___

[![Traffic-Signs-Recognition-for-Turkey](https://github.com/kemalkilicaslan/Traffic-Signs-Recognition-for-Turkey/blob/main/Traffic-Signs-Recognition-for-Turkey.jpg)](https://www.youtube.com/watch?v=TycwyddkcHQ)


[![Traffic-Signs-Recognition-for-Turkey-processed](https://github.com/kemalkilicaslan/Traffic-Signs-Recognition-for-Turkey/blob/main/Traffic-Signs-Recognition-for-Turkey-processed.jpg)](https://www.youtube.com/watch?v=N08fsgVGsdI)

___

![Classes](https://github.com/kemalkilicaslan/Traffic-Signs-Recognition-for-Turkey/blob/main/Classes.jpg)

![Dataset](https://github.com/kemalkilicaslan/Traffic-Signs-Recognition-for-Turkey/blob/main/Dataset.jpg)

![Health Check](https://github.com/kemalkilicaslan/Traffic-Signs-Recognition-for-Turkey/blob/main/Health-Check.gif)

![Versions](https://github.com/kemalkilicaslan/Traffic-Signs-Recognition-for-Turkey/blob/main/Versions.jpg)

![Dataset Details](https://github.com/kemalkilicaslan/Traffic-Signs-Recognition-for-Turkey/blob/main/Dataset-Details.jpg)

![Training Graphs](https://github.com/kemalkilicaslan/Traffic-Signs-Recognition-for-Turkey/blob/main/Training-Graphs.jpg)

___

# Traffic_Signs_Recognition_for_Turkey.ipynb

A list of 17 frequently encountered traffic identification signs in urban areas in Turkey

- bumpy road
- bus stop
- give way
- green light
- no left turn
- no right turn
- no u turn
- parking
- parking is forbidden
- pedestrian crossing
- red light
- road with no entrance
- school crossing
- stop
- stopping and parking is forbidden
- vehicle is towed
- yellow light



It is an individual project that I have developed to be used in real time for the control and automation of autonomous vehicles with the model I trained on 668 images using the [roboflow](https://universe.roboflow.com/kemalkilicaslan-zdxtt/trafficsignsrecognitionforturkey-enyxp) platform and video processing with [ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8x.

### Install YOLOv8


```python
!pip install ultralytics
```

    Collecting ultralytics
      Downloading ultralytics-8.1.29-py3-none-any.whl (721 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m721.3/721.3 kB[0m [31m11.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: matplotlib>=3.3.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (3.7.1)
    Requirement already satisfied: opencv-python>=4.6.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (4.8.0.76)
    Requirement already satisfied: pillow>=7.1.2 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (9.4.0)
    Requirement already satisfied: pyyaml>=5.3.1 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (6.0.1)
    Requirement already satisfied: requests>=2.23.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (2.31.0)
    Requirement already satisfied: scipy>=1.4.1 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (1.11.4)
    Requirement already satisfied: torch>=1.8.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (2.2.1+cu121)
    Requirement already satisfied: torchvision>=0.9.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (0.17.1+cu121)
    Requirement already satisfied: tqdm>=4.64.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (4.66.2)
    Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from ultralytics) (5.9.5)
    Requirement already satisfied: py-cpuinfo in /usr/local/lib/python3.10/dist-packages (from ultralytics) (9.0.0)
    Collecting thop>=0.1.1 (from ultralytics)
      Downloading thop-0.1.1.post2209072238-py3-none-any.whl (15 kB)
    Requirement already satisfied: pandas>=1.1.4 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (1.5.3)
    Requirement already satisfied: seaborn>=0.11.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (0.13.1)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (4.49.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (1.4.5)
    Requirement already satisfied: numpy>=1.20 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (1.25.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (24.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (3.1.2)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.1.4->ultralytics) (2023.4)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (2024.2.2)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (3.13.1)
    Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (4.10.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (3.2.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (3.1.3)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (2023.6.0)
    Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m23.7/23.7 MB[0m [31m58.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cuda-runtime-cu12==12.1.105 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m823.6/823.6 kB[0m [31m66.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cuda-cupti-cu12==12.1.105 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m14.1/14.1 MB[0m [31m60.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cudnn-cu12==8.9.2.26 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m731.7/731.7 MB[0m [31m2.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cublas-cu12==12.1.3.1 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m410.6/410.6 MB[0m [31m2.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cufft-cu12==11.0.2.54 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m121.6/121.6 MB[0m [31m8.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-curand-cu12==10.3.2.106 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m56.5/56.5 MB[0m [31m11.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cusolver-cu12==11.4.5.107 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m124.2/124.2 MB[0m [31m8.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cusparse-cu12==12.1.0.106 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m196.0/196.0 MB[0m [31m6.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-nccl-cu12==2.19.3 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_nccl_cu12-2.19.3-py3-none-manylinux1_x86_64.whl (166.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m166.0/166.0 MB[0m [31m7.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-nvtx-cu12==12.1.105 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m99.1/99.1 kB[0m [31m15.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: triton==2.2.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (2.2.0)
    Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch>=1.8.0->ultralytics)
      Downloading nvidia_nvjitlink_cu12-12.4.99-py3-none-manylinux2014_x86_64.whl (21.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.1/21.1 MB[0m [31m82.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics) (1.16.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.8.0->ultralytics) (2.1.5)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.8.0->ultralytics) (1.3.0)
    Installing collected packages: nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, thop, ultralytics
    Successfully installed nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.19.3 nvidia-nvjitlink-cu12-12.4.99 nvidia-nvtx-cu12-12.1.105 thop-0.1.1.post2209072238 ultralytics-8.1.29



```python
from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
display.clear_output()
!yolo checks
```

    Ultralytics YOLOv8.1.29 🚀 Python-3.10.12 torch-2.2.1+cu121 CUDA:0 (Tesla T4, 15102MiB)
    Setup complete ✅ (2 CPUs, 12.7 GB RAM, 28.9/78.2 GB disk)
    
    OS                  Linux-6.1.58+-x86_64-with-glibc2.35
    Environment         Colab
    Python              3.10.12
    Install             pip
    RAM                 12.67 GB
    CPU                 Intel Xeon 2.00GHz
    CUDA                12.1
    
    matplotlib          ✅ 3.7.1>=3.3.0
    opencv-python       ✅ 4.8.0.76>=4.6.0
    pillow              ✅ 9.4.0>=7.1.2
    pyyaml              ✅ 6.0.1>=5.3.1
    requests            ✅ 2.31.0>=2.23.0
    scipy               ✅ 1.11.4>=1.4.1
    torch               ✅ 2.2.1+cu121>=1.8.0
    torchvision         ✅ 0.17.1+cu121>=0.9.0
    tqdm                ✅ 4.66.2>=4.64.0
    psutil              ✅ 5.9.5
    py-cpuinfo          ✅ 9.0.0
    thop                ✅ 0.1.1-2209072238>=0.1.1
    pandas              ✅ 1.5.3>=1.1.4
    seaborn             ✅ 0.13.1>=0.11.0


### YOLOv8 Model Trained on Custom Dataset


```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("kemalkilicaslan-zdxtt").project("trafficsignsrecognitionforturkey-enyxp")
dataset = project.version(1).download("yolov8")
```

    Collecting roboflow
      Downloading roboflow-1.1.24-py3-none-any.whl (71 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m71.7/71.7 kB[0m [31m3.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting certifi==2023.7.22 (from roboflow)
      Downloading certifi-2023.7.22-py3-none-any.whl (158 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m158.3/158.3 kB[0m [31m10.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting chardet==4.0.0 (from roboflow)
      Downloading chardet-4.0.0-py2.py3-none-any.whl (178 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m178.7/178.7 kB[0m [31m14.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting cycler==0.10.0 (from roboflow)
      Downloading cycler-0.10.0-py2.py3-none-any.whl (6.5 kB)
    Collecting idna==2.10 (from roboflow)
      Downloading idna-2.10-py2.py3-none-any.whl (58 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.8/58.8 kB[0m [31m7.5 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.4.5)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from roboflow) (3.7.1)
    Requirement already satisfied: numpy>=1.18.5 in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.25.2)
    Collecting opencv-python-headless==4.8.0.74 (from roboflow)
      Downloading opencv_python_headless-4.8.0.74-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (49.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.1/49.1 MB[0m [31m11.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: Pillow>=7.1.2 in /usr/local/lib/python3.10/dist-packages (from roboflow) (9.4.0)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.8.2)
    Collecting python-dotenv (from roboflow)
      Downloading python_dotenv-1.0.1-py3-none-any.whl (19 kB)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.31.0)
    Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.16.0)
    Requirement already satisfied: urllib3>=1.26.6 in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.0.7)
    Requirement already satisfied: tqdm>=4.41.0 in /usr/local/lib/python3.10/dist-packages (from roboflow) (4.66.2)
    Requirement already satisfied: PyYAML>=5.3.1 in /usr/local/lib/python3.10/dist-packages (from roboflow) (6.0.1)
    Collecting requests-toolbelt (from roboflow)
      Downloading requests_toolbelt-1.0.0-py2.py3-none-any.whl (54 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m54.5/54.5 kB[0m [31m8.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting python-magic (from roboflow)
      Downloading python_magic-0.4.27-py2.py3-none-any.whl (13 kB)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (1.2.0)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (4.49.0)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (24.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (3.1.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->roboflow) (3.3.2)
    Installing collected packages: python-magic, python-dotenv, opencv-python-headless, idna, cycler, chardet, certifi, requests-toolbelt, roboflow
      Attempting uninstall: opencv-python-headless
        Found existing installation: opencv-python-headless 4.9.0.80
        Uninstalling opencv-python-headless-4.9.0.80:
          Successfully uninstalled opencv-python-headless-4.9.0.80
      Attempting uninstall: idna
        Found existing installation: idna 3.6
        Uninstalling idna-3.6:
          Successfully uninstalled idna-3.6
      Attempting uninstall: cycler
        Found existing installation: cycler 0.12.1
        Uninstalling cycler-0.12.1:
          Successfully uninstalled cycler-0.12.1
      Attempting uninstall: chardet
        Found existing installation: chardet 5.2.0
        Uninstalling chardet-5.2.0:
          Successfully uninstalled chardet-5.2.0
      Attempting uninstall: certifi
        Found existing installation: certifi 2024.2.2
        Uninstalling certifi-2024.2.2:
          Successfully uninstalled certifi-2024.2.2
    Successfully installed certifi-2023.7.22 chardet-4.0.0 cycler-0.10.0 idna-2.10 opencv-python-headless-4.8.0.74 python-dotenv-1.0.1 python-magic-0.4.27 requests-toolbelt-1.0.0 roboflow-1.1.24




    loading Roboflow workspace...
    loading Roboflow project...
    Dependency ultralytics==8.0.196 is required but found version=8.1.29, to fix: `pip install ultralytics==8.0.196`


    Downloading Dataset Version Zip in TrafficSignsRecognitionforTurkey-1 to yolov8:: 100%|██████████| 118577/118577 [00:02<00:00, 53083.60it/s]

    


    
    Extracting Dataset Version Zip to TrafficSignsRecognitionforTurkey-1 in yolov8:: 100%|██████████| 3130/3130 [00:00<00:00, 4505.75it/s]


### Load the YOLOv8x model and train model



```python
!yolo task=detect mode=train model=yolov8x.pt data={dataset.location}/data.yaml epochs=20 imgsz=640
```

    Downloading https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt to 'yolov8x.pt'...
    100% 131M/131M [00:00<00:00, 310MB/s]
    Ultralytics YOLOv8.1.29 🚀 Python-3.10.12 torch-2.2.1+cu121 CUDA:0 (Tesla T4, 15102MiB)
    [34m[1mengine/trainer: [0mtask=detect, mode=train, model=yolov8x.pt, data=/content/TrafficSignsRecognitionforTurkey-1/data.yaml, epochs=20, time=None, patience=100, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=8, project=None, name=train, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train
    Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf'...
    100% 755k/755k [00:00<00:00, 24.2MB/s]
    2024-03-16 23:18:51.796144: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-03-16 23:18:51.796241: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-03-16 23:18:51.910525: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    Overriding model.yaml nc=80 with nc=17
    
                       from  n    params  module                                       arguments                     
      0                  -1  1      2320  ultralytics.nn.modules.conv.Conv             [3, 80, 3, 2]                 
      1                  -1  1    115520  ultralytics.nn.modules.conv.Conv             [80, 160, 3, 2]               
      2                  -1  3    436800  ultralytics.nn.modules.block.C2f             [160, 160, 3, True]           
      3                  -1  1    461440  ultralytics.nn.modules.conv.Conv             [160, 320, 3, 2]              
      4                  -1  6   3281920  ultralytics.nn.modules.block.C2f             [320, 320, 6, True]           
      5                  -1  1   1844480  ultralytics.nn.modules.conv.Conv             [320, 640, 3, 2]              
      6                  -1  6  13117440  ultralytics.nn.modules.block.C2f             [640, 640, 6, True]           
      7                  -1  1   3687680  ultralytics.nn.modules.conv.Conv             [640, 640, 3, 2]              
      8                  -1  3   6969600  ultralytics.nn.modules.block.C2f             [640, 640, 3, True]           
      9                  -1  1   1025920  ultralytics.nn.modules.block.SPPF            [640, 640, 5]                 
     10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     12                  -1  3   7379200  ultralytics.nn.modules.block.C2f             [1280, 640, 3]                
     13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     15                  -1  3   1948800  ultralytics.nn.modules.block.C2f             [960, 320, 3]                 
     16                  -1  1    922240  ultralytics.nn.modules.conv.Conv             [320, 320, 3, 2]              
     17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     18                  -1  3   7174400  ultralytics.nn.modules.block.C2f             [960, 640, 3]                 
     19                  -1  1   3687680  ultralytics.nn.modules.conv.Conv             [640, 640, 3, 2]              
     20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     21                  -1  3   7379200  ultralytics.nn.modules.block.C2f             [1280, 640, 3]                
     22        [15, 18, 21]  1   8734339  ultralytics.nn.modules.head.Detect           [17, [320, 640, 640]]         
    Model summary: 365 layers, 68168979 parameters, 68168963 gradients, 258.2 GFLOPs
    
    Transferred 589/595 items from pretrained weights
    [34m[1mTensorBoard: [0mStart with 'tensorboard --logdir runs/detect/train', view at http://localhost:6006/
    Freezing layer 'model.22.dfl.conv.weight'
    [34m[1mAMP: [0mrunning Automatic Mixed Precision (AMP) checks with YOLOv8n...
    Downloading https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt to 'yolov8n.pt'...
    100% 6.23M/6.23M [00:00<00:00, 108MB/s]
    [34m[1mAMP: [0mchecks passed ✅
    [34m[1mtrain: [0mScanning /content/TrafficSignsRecognitionforTurkey-1/train/labels... 1351 images, 0 backgrounds, 0 corrupt: 100% 1351/1351 [00:00<00:00, 1709.48it/s]
    [34m[1mtrain: [0mNew cache created: /content/TrafficSignsRecognitionforTurkey-1/train/labels.cache
    WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 2926, len(boxes) = 3109. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
    [34m[1mval: [0mScanning /content/TrafficSignsRecognitionforTurkey-1/valid/labels... 137 images, 0 backgrounds, 0 corrupt: 100% 137/137 [00:00<00:00, 550.44it/s]
    [34m[1mval: [0mNew cache created: /content/TrafficSignsRecognitionforTurkey-1/valid/labels.cache
    WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 291, len(boxes) = 327. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
    Plotting labels to runs/detect/train/labels.jpg... 
    [34m[1moptimizer:[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
    [34m[1moptimizer:[0m AdamW(lr=0.000476, momentum=0.9) with parameter groups 97 weight(decay=0.0), 104 weight(decay=0.0005), 103 bias(decay=0.0)
    [34m[1mTensorBoard: [0mmodel graph visualization added ✅
    Image sizes 640 train, 640 val
    Using 2 dataloader workers
    Logging results to [1mruns/detect/train[0m
    Starting training for 20 epochs...
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           1/20      14.3G       1.25      3.832      1.172         31        640: 100% 85/85 [01:40<00:00,  1.18s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:05<00:00,  1.08s/it]
                       all        137        327      0.439      0.329      0.292      0.217
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           2/20      14.9G      1.222      2.495      1.177         22        640: 100% 85/85 [01:40<00:00,  1.18s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.13it/s]
                       all        137        327      0.598      0.287      0.322      0.228
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           3/20      14.9G      1.257      2.109      1.194         20        640: 100% 85/85 [01:41<00:00,  1.19s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.15it/s]
                       all        137        327      0.502      0.406       0.39      0.244
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           4/20      14.9G      1.238      1.922      1.171         23        640: 100% 85/85 [01:40<00:00,  1.18s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.11it/s]
                       all        137        327      0.536      0.407      0.403      0.281
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           5/20      14.9G       1.24       1.87      1.159         24        640: 100% 85/85 [01:41<00:00,  1.19s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.12it/s]
                       all        137        327      0.568       0.43      0.447      0.314
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           6/20      14.9G      1.197      1.714      1.147         14        640: 100% 85/85 [01:41<00:00,  1.19s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.14it/s]
                       all        137        327      0.501      0.443      0.451      0.314
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           7/20      14.9G      1.151      1.613      1.114         23        640: 100% 85/85 [01:41<00:00,  1.19s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.12it/s]
                       all        137        327      0.582      0.474      0.458       0.33
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           8/20      14.9G      1.115      1.562      1.096         23        640: 100% 85/85 [01:41<00:00,  1.19s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.16it/s]
                       all        137        327      0.622      0.438      0.456      0.333
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           9/20      14.9G      1.061      1.479      1.062         11        640: 100% 85/85 [01:41<00:00,  1.19s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.10it/s]
                       all        137        327      0.631      0.471      0.472      0.326
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          10/20      14.9G       1.05      1.404      1.066         12        640: 100% 85/85 [01:41<00:00,  1.20s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.15it/s]
                       all        137        327      0.628      0.509      0.496      0.362
    Closing dataloader mosaic
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          11/20      14.9G      1.053      1.326      1.045         15        640: 100% 85/85 [01:41<00:00,  1.20s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.12it/s]
                       all        137        327      0.681      0.463      0.506      0.366
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          12/20      14.9G      1.009      1.206      1.014         13        640: 100% 85/85 [01:41<00:00,  1.19s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.13it/s]
                       all        137        327      0.668      0.455        0.5      0.345
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          13/20      14.9G     0.9803      1.108     0.9855         10        640: 100% 85/85 [01:41<00:00,  1.19s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.13it/s]
                       all        137        327      0.716      0.458      0.527      0.368
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          14/20      14.9G     0.9406      1.047     0.9804         16        640: 100% 85/85 [01:40<00:00,  1.18s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.14it/s]
                       all        137        327      0.648      0.513      0.541      0.394
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          15/20      14.9G     0.9234     0.9925     0.9706         15        640: 100% 85/85 [01:40<00:00,  1.18s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.15it/s]
                       all        137        327       0.64      0.518      0.545      0.399
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          16/20      14.9G     0.8907     0.9617     0.9549          9        640: 100% 85/85 [01:40<00:00,  1.18s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.14it/s]
                       all        137        327      0.718      0.477      0.545      0.398
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          17/20      14.9G     0.8536     0.8838     0.9392         12        640: 100% 85/85 [01:41<00:00,  1.19s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.12it/s]
                       all        137        327      0.749      0.493      0.551      0.414
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          18/20      14.9G     0.8316     0.8319     0.9289          7        640: 100% 85/85 [01:40<00:00,  1.18s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.16it/s]
                       all        137        327      0.675      0.524      0.548      0.407
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          19/20      14.9G     0.8152     0.7967     0.9205         18        640: 100% 85/85 [01:41<00:00,  1.19s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.17it/s]
                       all        137        327      0.679      0.509      0.553      0.412
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          20/20      14.9G     0.7806     0.7389     0.9101         12        640: 100% 85/85 [01:41<00:00,  1.19s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.16it/s]
                       all        137        327      0.767      0.488      0.561      0.423
    
    20 epochs completed in 0.660 hours.
    Optimizer stripped from runs/detect/train/weights/last.pt, 136.7MB
    Optimizer stripped from runs/detect/train/weights/best.pt, 136.7MB
    
    Validating runs/detect/train/weights/best.pt...
    Ultralytics YOLOv8.1.29 🚀 Python-3.10.12 torch-2.2.1+cu121 CUDA:0 (Tesla T4, 15102MiB)
    Model summary (fused): 268 layers, 68139939 parameters, 0 gradients, 257.5 GFLOPs
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:05<00:00,  1.18s/it]
                       all        137        327      0.735      0.495      0.561      0.423
                bumpy road        137          5      0.528        0.4      0.505      0.414
                  bus stop        137         17       0.83      0.765      0.819      0.607
                  give way        137          5      0.755        0.4      0.449      0.378
               green light        137         62       0.72      0.484      0.557      0.369
              no left turn        137         17      0.634       0.51      0.529      0.415
             no right turn        137         16      0.663        0.5      0.549      0.391
                 no u turn        137         15      0.925      0.733      0.798       0.69
                   parking        137         16      0.711      0.438      0.453      0.335
      parking is forbidden        137         19      0.737      0.684       0.74      0.619
       pedestrian crossing        137         30      0.695      0.467      0.477      0.342
                 red light        137         30      0.655      0.133      0.237      0.157
     road with no entrance        137         11      0.656     0.0909       0.12      0.108
           school crossing        137         19      0.669      0.632       0.68      0.488
                      stop        137          5      0.738        0.6      0.735      0.529
    stopping and parking is forbidden        137         26      0.947      0.687      0.804      0.633
          vehicle is towed        137         15      0.764      0.533      0.662      0.467
              yellow light        137         19      0.874      0.367      0.423      0.256
    Speed: 0.2ms preprocess, 26.1ms inference, 0.0ms loss, 2.6ms postprocess per image
    Results saved to [1mruns/detect/train[0m
    💡 Learn more at https://docs.ultralytics.com/modes/train


### Evaluation


```python
# Confusion matrix

Image(filename=f"/content/runs/detect/train/confusion_matrix.png", width=600)
```




    
![Confusion matrix](https://github.com/kemalkilicaslan/Traffic-Signs-Recognition-for-Turkey/blob/main/Confusion-matrix.jpg)
    




```python
Image(filename="/content/runs/detect/train/results.png", width=600)
```




    
![results](https://github.com/kemalkilicaslan/Traffic-Signs-Recognition-for-Turkey/blob/main/results.jpg)
    




```python
!yolo task=detect mode=val model=/content/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
```

    Ultralytics YOLOv8.1.29 🚀 Python-3.10.12 torch-2.2.1+cu121 CUDA:0 (Tesla T4, 15102MiB)
    Model summary (fused): 268 layers, 68139939 parameters, 0 gradients, 257.5 GFLOPs
    [34m[1mval: [0mScanning /content/TrafficSignsRecognitionforTurkey-1/valid/labels.cache... 137 images, 0 backgrounds, 0 corrupt: 100% 137/137 [00:00<?, ?it/s]
    WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 291, len(boxes) = 327. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 9/9 [00:10<00:00,  1.16s/it]
                       all        137        327      0.735      0.496      0.562       0.43
                bumpy road        137          5      0.528        0.4      0.505      0.414
                  bus stop        137         17       0.83      0.765      0.819      0.607
                  give way        137          5      0.753        0.4      0.449      0.378
               green light        137         62       0.72      0.484      0.557       0.37
              no left turn        137         17      0.634       0.51      0.558      0.422
             no right turn        137         16      0.663        0.5      0.549      0.405
                 no u turn        137         15      0.926      0.733      0.798      0.695
                   parking        137         16      0.714      0.438      0.453      0.362
      parking is forbidden        137         19      0.736      0.684      0.737      0.601
       pedestrian crossing        137         30      0.696      0.467      0.477      0.349
                 red light        137         30      0.654      0.133      0.237      0.169
     road with no entrance        137         11      0.656     0.0909       0.12     0.0983
           school crossing        137         19      0.669      0.632      0.677      0.498
                      stop        137          5      0.738        0.6      0.735      0.569
    stopping and parking is forbidden        137         26      0.947      0.687      0.803       0.65
          vehicle is towed        137         15      0.765      0.533      0.662      0.467
              yellow light        137         19      0.875      0.367       0.42      0.256
    Speed: 1.5ms preprocess, 57.4ms inference, 0.1ms loss, 6.7ms postprocess per image
    Results saved to [1mruns/detect/val[0m
    💡 Learn more at https://docs.ultralytics.com/modes/val

___

# Traffic_Signs_Recognition_for_Turkey.py

```python
from ultralytics import YOLO

model = YOLO('Traffic_Signs_Recognition_for_Turkey.pt')
results = model('Traffic_Signs_Recognition_for_Turkey.mp4', save=True)
```