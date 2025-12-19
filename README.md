# Coronary_Plaque_Detection
The official implementation of 'Cross-sectional angle prediction of lipid-rich and calcified tissue on computed tomography angiography images'.

## Dockerfile
You can simply build the inference image in a WSL2 environment using the Dockerfile in [Cartesian/Dockerfile](Cartesian/Dockerfile/) (coronary plaque detection in a Cartesian view using a 2.5D Dense U-Net), 
or using the Dockerfile in [Polar/Dockerfile](Polar/Dockerfile/) (coronary plaque detection in a Polar view using a 2.5D Mask R-CNN).
```
cd Cartesian/Dockerfile 
docker build -t plaque_det:cartesian .
## In the run.sh, replace the path in '-v /mnt/e/WSL/TestData/PlaqueDet/oriCT/deeplearning:/data_test_CT \' with your own data path.
bash run.sh
```
```
cd Polar/Dockerfile 
docker build -t plaque_det:polar .
## In the run.sh, replace the path in '-v /mnt/e/WSL/TestData/PlaqueDet/oriCT/deeplearning:/data_test_CT \' with your own data path.
bash run.sh
```
