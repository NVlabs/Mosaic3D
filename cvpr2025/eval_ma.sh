# evaluate our (spunet34C + ppt + sc+ar+ma+st+sc++, jobid = 3940849)
# python src/eval.py experiment=train_spunet_multidata_ppt data=sc+ar+ma+st+sc++_val_ma ckpt_path=results/3940849/checkpoints/epoch_118.ckpt
# python src/eval.py \
#     experiment=train_spunet_multidata_ppt \
#     data=sc+ar+ma+st+sc++_test_ma \
#     ckpt_path=results/3940849/checkpoints/epoch_118.ckpt \
#     logger=csv \
#     logger.csv.save_dir=./eval_matterport3d/spunet34c+ppt+all

# evaluate eth3d
SAVE_PRED=true python src/eval.py \
    experiment=train_spunet_multidata_ppt \
    data=eth3d \
    ckpt_path=results/3940849/checkpoints/epoch_118.ckpt \
    logger=csv \
    logger.csv.save_dir=./eval_eth3d/spunet34c+ppt+all

# python src/eval.py \
#     experiment=train_regionplc_scannet \
#     data=matterport3d \
#     model.net.backbone_cfg.mid_channel=32 \
#     model.use_prompt=true \
#     model.net.adapter_cfg.last_norm=true \
#     ckpt_path=/lustre/fsw/portfolios/nvr/users/chunghyunp/projects/openvocab-3d/results/4001966/checkpoints/epoch_124.ckpt\
#     logger=csv \
#     logger.csv.save_dir=./eval_scannet/regionplc

# python src/eval.py \
#     experiment=train_spunet_multidata_ppt \
#     data=sc_test_ma \
#     ckpt_path=results/3988223/checkpoints/epoch_126.ckpt \
#     logger=csv \
#     logger.csv.save_dir=./eval_scannet/spunet34c+ppt+sc

# evaluate openscene
# python src/eval.py experiment=train_spunet_scannet data=matterport3d model=minkunet18a ckpt_path=ckpts/matterport_lseg.ckpt model/clip=clip-b32 +data.collate_fn.drop_feat=true

# python src/eval.py \
#  experiment=train_spunet_scannet \
#  model=minkunet18a \
#  ckpt_path=ckpts/matterport_lseg.ckpt \
#  model/clip=clip-b32 \
#  +data.collate_fn.drop_feat=true \
#  data=matterport3d \
#  logger=csv \
#  logger.csv.save_dir=./eval_matterport3d/openscene_lseg

# python src/eval.py \
#  experiment=train_spunet_scannet \
#  model=minkunet18a \
#  ckpt_path=ckpts/matterport_openseg.ckpt \
#  model/clip=clip-l14 \
#  +data.collate_fn.drop_feat=true \
#  data=matterport3d \
#  logger=csv \
#  logger.csv.save_dir=./eval_matterport3d/openscene_openseg

# python src/eval.py \
#  experiment=train_spunet_scannet \
#  model=minkunet18a \
#  ckpt_path=ckpts/openscene_lseg.ckpt \
#  model/clip=clip-b32 \
#  +data.collate_fn.drop_feat=true \
#  data=matterport3d \
#  logger=csv \
#  logger.csv.save_dir=./eval_matterport3d/openscene_lseg_scannet

# python src/eval.py \
#  experiment=train_spunet_scannet \
#  model=minkunet18a \
#  ckpt_path=ckpts/openscene_openseg.ckpt \
#  model/clip=clip-l14 \
#  +data.collate_fn.drop_feat=true \
#  data=matterport3d \
#  logger=csv \
#  logger.csv.save_dir=./eval_matterport3d/openscene_openseg_scannet