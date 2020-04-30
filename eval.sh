#CUDA_VISIBLE_DEVICES=0,1 python projects/EDNet_seg/train_net.py --config-file projects/EDNet_seg/configs/deeplab_det.yaml --eval-only MODEL.WEIGHTS output/ednet_dl_det_0/model_final.pth
CUDA_VISIBLE_DEVICES=0,1 python projects/EDNet_seg/train_net.py --config-file projects/EDNet_seg/configs/seg_det.yaml --eval-only MODEL.WEIGHTS output/ednet_seg_det_12/model_final.pth
#CUDA_VISIBLE_DEVICES=0,1 python projects/EDNet_v1/train_net.py --config-file projects/EDNet_v1/configs/same_ednet.yaml --eval-only MODEL.WEIGHTS output/ednet_v1/model_final.pth
#CUDA_VISIBLE_DEVICES=3 python projects/EDNet/train_net.py --config-file projects/EDNet/configs/same_ednet.yaml --eval-only MODEL.WEIGHTS output/ednet_v2_mulgpu/model_final.pth
