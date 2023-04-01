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
