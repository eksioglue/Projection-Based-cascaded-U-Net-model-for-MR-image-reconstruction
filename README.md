# Projection-Based-cascaded-U-Net-model-for-MR-image-reconstruction
[Dataset](https://fastmri.med.nyu.edu/) 
* **Article:** [Projection-Based cascaded U-Net model for MR image reconstruction ({A. Aghabiglou, E.M. Eksioglu}, 2021)](https://www.sciencedirect.com/science/article/abs/pii/S016926072100225X)


## Training a model

```bash
python train_unet.py --mode train  --challenge singlecoil --data_path "/Path/to/Dataset/" --exp Unet_4x --num-epochs 20 --gpus 2 --batch_size 16 --lr 0.0001  --center-fractions 0.08 --accelerations 4
```
```bash
python train_cascade_unet.py --mode train  --challenge singlecoil --data_path "/Path/to/Dataset/" --exp cascade_Unet_4x --num-epochs 20 --gpus 2 --batch_size 4 --lr 0.0001  --center-fractions 0.08 --accelerations 4
```
## Test

```bash
python train_cascade_unet.py --mode test  --challenge singlecoil --data_path "/Path/to/Dataset/"  --checkpoint  "/Path/to/checkpoint/"   --exp '/PATH/TO/SAVE/FOLDER/' --scname_test validationset_fullnumpy   --rec2_ext G1_cascade_unet8x
```

## Cite

If you use the code in your project, please cite the paper:

```BibTeX
@article{AGHABIGLOU2021106151,
title = {Projection-Based cascaded U-Net model for MR image reconstruction},
journal = {Computer Methods and Programs in Biomedicine},
volume = {207},
pages = {106151},
year = {2021},
issn = {0169-2607},
doi = {https://doi.org/10.1016/j.cmpb.2021.106151},
url = {https://www.sciencedirect.com/science/article/pii/S016926072100225X},
author = {Amir Aghabiglou and Ender M. Eksioglu},
keywords = {Magnetic resonance imaging, Image reconstruction, Deep learning, Cascaded networks, U-Net, Updated data consistency}
}
```
