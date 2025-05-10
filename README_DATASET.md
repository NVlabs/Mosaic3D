# Mosaic3D++ Dataset Preparation

## Dataset

Currently, this codebase support training and evaluation on the official dataset provided by [RegionPLC](https://github.com/CVMI-Lab/PLA/tree/regionplc).

To download the dataset, use the following command:

| \*Note that you should update the `FedAuth` token in `src/data/regionplc/download.py` script to recently generated one before downloading.

```bash
python -m src.data.regionplc.download --download_dir [path/to/save/dataset]
# e.g. python -m src.data.region.download --download_dir /home/junhal/datasets/regionplc
```

#### How to obtain fresh `FedAuth` token?

<details><summary>Click</summary>

1. Open the following [link](https://connecthkuhk-my.sharepoint.com/personal/jhyang13_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjhyang13%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fpretrained%5Fmodels%2Fregionplc%2Fcaption%5Ffiles%2Fscannet&ga=1) on Google Chrome.

2. Open developer tool and head to the Network tab. Apply `Doc` filter

3. Click a checkbox of some large file, such as `scannet_caption_idx_detic_and_sw_125k_iou0.3.pkl`

4. Click Download button above

<!-- ![alt text](assets/image.png) -->

<img src="assets/image.png" height="400">

5. Cancel the download process

6. Search for download payload on the Network tab

7. Click right mouse button and click `Copy as cURL`

<!-- ![alt text](assets/image-1.png) -->

<img src='assets/image-1.png' height="400">

08. Paste the copied cURL command somewhere and search for `FedAuth` token.

09. Update `FedAuth` variable in `src/data/regionplc/download.py` to new one.

10. Execute the above download command.

</details>
