# SSP3D

# Set ShapeNet datasets
imagefile  dataset/ShapeNetRendering
voxelfile  dataset/ShapeNetVox32

# Set Pix3D dataset
imagefile dataset/img/
voxelfile dataset/model/


# train ShapeNet stage1
python runner_shapenet.py

# train ShapeNet stage2
python runner_shapenet.py --finetune --weights=xxx.pth


# test ShapeNet
python runner_shapenet.py --test --weights=xxx.pth


# train Pix3D stage1
python runner_pix3d.py

# train Pix3D stage2
python runner_pix3d.py --finetune --weights=xxx.pth

# test Pix3D 
python runner_pix3d.py --test --weights=xxx.pth
