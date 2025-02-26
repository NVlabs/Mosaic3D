export HF_HUB_CACHE=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_nvgptvision/huggingface_cache

# RegionPLC32 + CLIP-B16
# # leo
# python src/eval.py \
#     experiment=train_regionplc_multidata \
#     model=regionplc32 \
#     model/clip=clip-b16 \
#     data=leo \
#     ckpt_path=results/4928121/checkpoints/epoch_108.ckpt \
#     logger=csv \
#     logger.csv.save_dir=./eval_rebuttal/regionplc32_leo

# # sceneverse
# python src/eval.py \
#     experiment=train_regionplc_multidata \
#     model=regionplc32 \
#     model/clip=clip-b16 \
#     data=sceneverse \
#     ckpt_path=results/4957764/checkpoints/epoch_104.ckpt \
#     logger=csv \
#     logger.csv.save_dir=./eval_rebuttal/regionplc32_sceneverse

# # mosaic3d
# python src/eval.py \
#     experiment=train_regionplc_multidata \
#     model=regionplc32 \
#     model/clip=clip-b16 \
#     data=sc+ar+ma+st+sc++ \
#     ckpt_path=results/4934668/checkpoints/epoch_105.ckpt \
#     logger=csv \
#     logger.csv.save_dir=./eval_rebuttal/regionplc32_mosaic3d

# # mosaic3d - scannet
# python src/eval.py \
#     experiment=train_spunet_multidata \
#     data=sc+ar+ma+st+sc++ \
#     ckpt_path=results/4934679/checkpoints/epoch_121.ckpt \
#     logger=csv \
#     logger.csv.save_dir=./eval_rebuttal/spunet_wo_scannet

# # mosaic3d - ma3d
# python src/eval.py \
#     experiment=train_spunet_multidata \
#     data=sc+ar+ma+st+sc++_test_ma \
#     ckpt_path=results/4950010/checkpoints/epoch_121.ckpt \
#     logger=csv \
#     logger.csv.save_dir=./eval_rebuttal/spunet_wo_ma3d


# SPUNet34C + RecapCLIP
# # leo
# python src/eval.py \
#     experiment=train_spunet_multidata \
#     data=leo \
#     ckpt_path=results/4993792/checkpoints/epoch_126.ckpt \
#     logger=csv \
#     logger.csv.save_dir=./eval_rebuttal/spunet34c_leo

# # sceneverse
# python src/eval.py \
#     experiment=train_spunet_multidata \
#     data=sceneverse \
#     ckpt_path=results/4993795/checkpoints/epoch_126.ckpt \
#     logger=csv \
#     logger.csv.save_dir=./eval_rebuttal/spunet34c_sceneverse

# mosaic3d
python src/eval.py \
    experiment=train_spunet_multidata_ppt \
    data=sc+ar+ma+st+sc++ \
    ckpt_path=results/3940849/checkpoints/epoch_118.ckpt \
    logger=csv \
    logger.csv.save_dir=./eval_rebuttal/spunet34c_mosaic3d
