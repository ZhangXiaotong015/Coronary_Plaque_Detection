# Coronary_Plaque_Detection
The official implementation of 'Cross-sectional angle prediction of lipid-rich and calcified tissue on computed tomography angiography images'.

## Dockerfile
You can simply build the inference image in a WSL2 environment using the Dockerfile in [Cartesian/Dockerfile](Cartesian/Dockerfile/) (coronary plaque detection in a Cartesian view using a 2.5D Dense U-Net), 
or using the Dockerfile in [Polar/Dockerfile](Polar/Dockerfile/) (coronary plaque detection in a Polar view using a 2.5D Mask R-CNN).
```
cd Cartesian/Dockerfile # or cd Polar/Dockerfile
docker build -t plaque_det:latest .
## In the run.sh, replace the src path in '--mount type=bind,src=/mnt/e/WSL/TestData/LiverVesselSeg/Pre-ablation/Portal,dst=/data_test_CT,readonly \' with your own data path.
bash run.sh
```
