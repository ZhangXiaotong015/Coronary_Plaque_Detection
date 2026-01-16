# Coronary_Plaque_Detection
The official implementation of 'Cross-sectional angle prediction of lipid-rich and calcified tissue on computed tomography angiography images'.

## Dockerfile
You can simply build the inference image in a WSL2 environment using the Dockerfile in [Cartesian/Dockerfile](Cartesian/Dockerfile/) (coronary plaque detection in a Cartesian view using a 2.5D Dense U-Net), 
or using the Dockerfile in [Polar/Dockerfile](Polar/Dockerfile/) (coronary plaque detection in a Polar view using a 2.5D Mask R-CNN).

**Cartesian view**
```
cd Cartesian/Dockerfile 
docker build -t plaque_det:cartesian .
## In the run.sh, replace the path in '-v /mnt/e/WSL/TestData/PlaqueDet/oriCT/deeplearning:/data_test_CT \' with your own data path.
bash run.sh
```
```Contents of the output folder:```

```/xxxxxx/xxx.nii.gz:``` Slice-wise probability distribution map of the straightened MPR.

```Contents of the output_chemogram folder:```

```/xxxxxx/xxx.png:``` PNG-formatted spread-out view of lipid-rich or calcified plaques.

```xxx.nii.gz:``` NITFI-formatted spread-out view of lipid-rich or calcified plaques.

**Polar view**
```
cd Polar/Dockerfile 
docker build -t plaque_det:polar .
## In the run.sh, replace the path in '-v /mnt/e/WSL/TestData/PlaqueDet/oriCT/deeplearning:/data_test_CT \' with your own data path.
bash run.sh
```
If you want to build inference image for the Cartesian view, you can find the model weights and the 'sample_line' files at [this link](https://drive.google.com/drive/folders/1yntV1aQUuT-v-DG_rPg5gq1kKEzP5bVv?usp=drive_link) and download them to ```Cartesian/Dockerfile/model```.

If you want to build inference image for the polar view, you can find the model weights and the 'sample_line' files at [this link](https://drive.google.com/drive/folders/1h_tOgkgco-ynV1f7dgLWVl0h892T40mp?usp=drive_link) and download them to ```Polar/Dockerfile/model```.

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
