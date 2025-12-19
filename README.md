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

## Apptainer/Singularity container system
If you have a Docker image built as mentioned above, you can save the Docker image to a ```.tar``` file and convert it to a ```SIF``` file, which is compatible with Apptainer.
For predictions in Cartesian view:
```
docker save -o plaque_det_denseunet.tar plaque_det:cartesian
```
You can use the bash file in [Cartesian/Apptainer](Cartesian/Apptainer/) to run the inference. 
```
cd Cartesian/Apptainer
bash bash_run_plaque_DenseUNet.sh
```

For predictions in polar view:
```
docker save -o bash_run_plaque_MaskRCNN.tar plaque_det:polar
```
You can use the bash file in [Polar/Apptainer](Polar/Apptainer/) to run the inference. 
```
cd Cartesian/Apptainer
bash bash_run_plaque_DenseUNet.sh
```
