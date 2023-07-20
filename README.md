## TractCloud

This repository releases the source code, training data, trained model, and testing samples for the work, "TractCloud: Registration-free tractography parcellation with a novel local-global streamline point cloud representation", which is accepted by MICCAI 2023.

![overview_v3](https://github.com/tengfeixue-victor/TractCloud-OpenSource/assets/56477109/1d41ef2c-367e-41dc-bfe2-6df955fc89d3)

## License

The contents of this repository are released under an [Slicer](LICENSE) license.

## Dependencies

The environment test was performed on RTX4090 and A5000

`conda create --name TractCloud python=3.8`

`conda activate TractCloud`

`conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch`

`conda install -c fvcore -c iopath -c conda-forge fvcore iopath`

`conda install -c bottler nvidiacub`

`pip install pytorch3d`

`pip install git+https://github.com/SlicerDMRI/whitematteranalysis.git`

`pip install h5py`

`pip install seaborn`

`pip install scikit-learn`

`pip install openpyxl`

## Training on anatomically curated atlas (ORG atlas)

The ORG atlas used in training is available at http://dmri.slicer.org/atlases/. You can directly download our processed data at https://github.com/SlicerDMRI/TractCloud/releases (1 million streamlines, 800 clusters & 800 outliers).
1. Download `TrainData_800clu800ol.zip` to `./` and `tar -xzvf TrainData_800clu800ol.zip`
2. Run `sh TrainOnAtlas.sh`

## Training on your custom dataset
Your input streamline features should have size of (number_streamlines, number_points_per_streamline, 3), and size of labels is (number_streamlines, ). You may save/load features and labels using .pickle files.

## Train/Validation/Test results and tips
The script calculates the accuracy and f1 on 42 anatomically meaningful tracts and one "Other" category (43 classes).

For training using the setting reported in our paper (k=20, k_global=500), most of CPU memory consumption comes from k. If you get out of CPU memory issue, you can try to reduce the value of k. Most of GPU memory consumption comes from k_global. If you get out of GPU memory issue, you can try to reduce the value of k_global.

## Testing on real data (registration-free parcellation)
Use the our trained model to parcellate real tractography data without registration.
1. Download `TrainedModel.zip` (https://github.com/SlicerDMRI/TractCloud/releases) to `./`, and `tar -xzvf TrainedModel.zip`
2. Download `TestData.zip` (https://github.com/SlicerDMRI/TractCloud/releases) to `./`, and `tar -xzvf TestData.zip`
3. Run `sh TractCloud.sh`

## Visualizing test parcellation results

Install 3D Slicer (https://www.slicer.org) and SlicerDMRI (http://dmri.slicer.org).

vtp/vtk files of 42 anatomically meaningful tracts are in `./parcellation_results/[test_data]/[subject_id]/SS/predictions`. "SS" means subject space. 

You can visualize them using 3D Slicer.

![TestExamples](https://github.com/SlicerDMRI/TractCloud/assets/56477109/5173e233-c66d-4954-b483-6fbf37b87310)


## References

**Please cite the following papers for using the code and/or the training data:**
    
    Tengfei Xue, Yuqian Chen, Chaoyi Zhang, Alexandra J. Golby, Nikos Makris, Yogesh Rathi, Weidong Cai, Fan Zhang, Lauren J. O'Donnell 
    TractCloud: Registration-free Tractography Parcellation with a Novel Local-global Streamline Point Cloud Representation.
    International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2023.

    Zhang, F., Wu, Y., Norton, I., Rathi, Y., Makris, N., O'Donnell, LJ. 
    An anatomically curated fiber clustering white matter atlas for consistent white matter tract parcellation across the lifespan. 
    NeuroImage, 2018 (179): 429-447

**For projects using Slicer and SlicerDMRI please also include the following text (or similar) and citations:**

* How to cite the [Slicer platform](http://wiki.slicer.org/slicerWiki/index.php/CitingSlicer)
* An example of how to cite SlicerDMRI (modify the first part of the sentence according to your use case):

    "We performed diffusion MRI tractography and/or analysis and/or visualization in 3D Slicer (www.slicer.org) via the SlicerDMRI project (dmri.slicer.org) (Norton et al. 2017)."
    
    Fan Zhang, Thomas Noh, Parikshit Juvekar, Sarah F Frisken, Laura Rigolo, Isaiah Norton, Tina Kapur, Sonia Pujol, William Wells III, Alex Yarmarkovich, Gordon Kindlmann, Demian Wassermann, Raul San Jose Estepar, Yogesh Rathi, Ron Kikinis, Hans J Johnson, Carl-Fredrik Westin, Steve Pieper, Alexandra J Golby, Lauren J O’Donnell. 
    SlicerDMRI: Diffusion MRI and Tractography Research Software for Brain Cancer Surgery Planning and Visualization. 
    JCO Clinical Cancer Informatics 4, e299-309, 2020.

    Isaiah Norton, Walid Ibn Essayed, Fan Zhang, Sonia Pujol, Alex Yarmarkovich, Alexandra J. Golby, Gordon Kindlmann, Demian Wassermann, Raul San Jose Estepar, Yogesh Rathi, Steve Pieper, Ron Kikinis, Hans J. Johnson, Carl-Fredrik Westin and Lauren J. O'Donnell. 
    SlicerDMRI: Open Source Diffusion MRI Software for Brain Cancer Research. Cancer Research 77(21), e101-e103, 2017.
