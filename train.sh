###v2
#CUDA_VISIBLE_DEVICES=0 python projects/EDNet/train_net.py --num-gpus 1 --config-file projects/EDNet/configs/Base-ednet.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.01
#CUDA_VISIBLE_DEVICES=1 python projects/EDNet/train_net.py --config-file projects/EDNet/configs/same_ednet.yaml
#CUDA_VISIBLE_DEVICES=0,1 python projects/EDNet_v2/train_net.py --num-gpus 2 --config-file projects/EDNet_v2/configs/same_ednet.yaml

##v2.1 
# Adjust the Warm-up step to be 5% of the whole training step
# Add a 1x1 conv after BiFPN to change the channel size from 64 to 256, so the channel size of head will be consistent with FPN. 
#CUDA_VISIBLE_DEVICES=0,1 python projects/EDNet_v2/train_net.py --num-gpus 2 --config-file projects/EDNet_v2/configs/same_ednet.yaml

###v1
#CUDA_VISIBLE_DEVICES=1,2,3 python projects/EDNet_v1/train_net.py --num-gpus 3 --config-file projects/EDNet_v1/configs/same_ednet.yaml
#CUDA_VISIBLE_DEVICES=0,1 python projects/EDNet_v1/train_net.py --num-gpus 2 --config-file projects/EDNet_v1/configs/same_ednet.yaml 
#python projects/EDNet_v1/train_net.py --num-gpus 0 --config-file projects/EDNet_v1/configs/same_ednet.yaml

# #seg_det 
# CUDA_VISIBLE_DEVICES=0,1 python projects/EDNet_seg/train_net.py --num-gpus 2 --config-file projects/EDNet_seg/configs/seg_det.yaml

#seg_det 
CUDA_VISIBLE_DEVICES=0,1 python projects/EDNet_seg/train_net.py --num-gpus 2 --config-file projects/EDNet_seg/configs/seg_det_keypoint.yaml

##deeplab_det 
#CUDA_VISIBLE_DEVICES=0,1 python projects/EDNet_seg/train_net.py --num-gpus 2 --config-file projects/EDNet_seg/configs/deeplab_det.yaml
