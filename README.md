# Vision Transformer-Based Semi-Supervised Semantic Segmentation with BG-CDF
This repository contains a Vision Transformer (ViT)-based semi-supervised semantic segmentation framework integrated with the Boundary-Guided Class Distribution Flow (BG-CDF) loss. The implementation is based on the [Segmenter](https://github.com/rstrudel/segmenter) framework and includes experiments on the Meta wildfire imagery dataset.
## Installation
Define environment variables pointing to your checkpoint and dataset directories, for example:
```bash
export DATASET=/path/to/dataset/dir
```
Install the required dependencies using:
```bash
pip install -r requirements.txt
```
You can also install the repository as a package from the root directory:
```bash
pip install .
```
## Model Zoo
We provide semi-supervised segmentation models trained on the Meta dataset using a Vision Transformer Tiny backbone (`vit_tiny_patch16_384`) with a Mask Transformer decoder and the BG-CDF loss.
### Meta Dataset
The Meta dataset uses a unified four-class taxonomy consisting of background, fire, burned area, and water.
| Model | Labeled Data | Unlabeled Data | Backbone | Decoder | Download |
| --- | --- | --- | --- | --- | --- |
| MODEL_FILE_0.4 | 40% | 60% | ViT-Tiny-Patch16-384 | Mask Transformer | [Model files](https://drive.google.com/drive/u/1/folders/1DmsfGvBoOKg93RGwIr7kCFrx3FPbMFir) |
| MODEL_FILE_0.5 | 50% | 50% | ViT-Tiny-Patch16-384 | Mask Transformer | [Model files](https://drive.google.com/drive/u/1/folders/1DmsfGvBoOKg93RGwIr7kCFrx3FPbMFir) |
| MODEL_FILE_0.6 | 60% | 40% | ViT-Tiny-Patch16-384 | Mask Transformer | [Model files](https://drive.google.com/drive/u/1/folders/1DmsfGvBoOKg93RGwIr7kCFrx3FPbMFir) |
| MODEL_FILE_0.7 | 70% | 30% | ViT-Tiny-Patch16-384 | Mask Transformer | [Model files](https://drive.google.com/drive/u/1/folders/1DmsfGvBoOKg93RGwIr7kCFrx3FPbMFir) |
Each model folder contains the trained checkpoint, configuration file, evaluation metrics, training losses, and training/evaluation plots.
### Meta Dataset Download
The Meta dataset used for the experiments is available here:
[Download the Meta dataset](https://drive.google.com/drive/u/1/folders/1cDXOGkIqONxV0w9yTdpo6sUfOpoPRJjZ)
## Inference
Download one of the trained checkpoints from the Model Zoo and place it in an appropriate folder.
For Meta dataset inference, define the dataset directory:
```bash
export DATASET=/path/to/Datasets/Meta
```
For example, to run inference using the 40% labeled-data model:
```bash
python3 inference.py \
    --model-path /path/to/MODEL_FILE_0.4/checkpoint.pth \
    --input-dir "$DATASET/images/test/" \
    --output-dir /path/to/PREDICTION_0.4/meta/ \
    --gt-dir "$DATASET/masks/test/"
```
The `--gt-dir` argument can be provided when ground-truth masks are available. The inference script generates segmentation predictions and evaluates them against the corresponding ground-truth masks.
The available labeled-data configurations are:
```text
--labeled-ratio 0.4  -> MODEL_FILE_0.4
--labeled-ratio 0.5  -> MODEL_FILE_0.5
--labeled-ratio 0.6  -> MODEL_FILE_0.6
--labeled-ratio 0.7  -> MODEL_FILE_0.7
```
## Train
### Meta Dataset Training
The following command trains the BG-CDF-enhanced semi-supervised model using 40% labeled data and 60% unlabeled data.
```bash
CUDA_VISIBLE_DEVICES=6,7 python3 train.py \
    --dataset-dir /path/to/Datasets/Meta/ \
    --teacher-dir /path/to/segmenter_supervised_META/segm/MODEL_FILE/ \
    --log-dir /path/to/segm/MODEL_FILE_0.4/ \
    --dataset meta \
    --backbone vit_tiny_patch16_384 \
    --decoder mask_transformer \
    --batch-size 4 \
    --epochs 50 \
    --learning-rate 0.001 \
    --labeled-ratio 0.4 \
    --eval-freq 1
```
The labeled-data ratio can be changed to train the other configurations:
```text
--labeled-ratio 0.4  -> MODEL_FILE_0.4
--labeled-ratio 0.5  -> MODEL_FILE_0.5
--labeled-ratio 0.6  -> MODEL_FILE_0.6
--labeled-ratio 0.7  -> MODEL_FILE_0.7
```
For example, the `0.7` configuration uses 70% labeled data and 30% unlabeled data:
```bash
CUDA_VISIBLE_DEVICES=6,7 python3 train.py \
    --dataset-dir /path/to/Datasets/Meta/ \
    --teacher-dir /path/to/segmenter_supervised_META/segm/MODEL_FILE/ \
    --log-dir /path/to/segm/MODEL_FILE_0.7/ \
    --dataset meta \
    --backbone vit_tiny_patch16_384 \
    --decoder mask_transformer \
    --batch-size 4 \
    --epochs 50 \
    --learning-rate 0.001 \
    --labeled-ratio 0.7 \
    --eval-freq 1
```
The training framework uses a supervised loss on labeled samples together with semi-supervised consistency learning on unlabeled samples. The BG-CDF loss provides an additional boundary-aware regularization term by comparing teacher and student class-probability profiles sampled across teacher-detected semantic boundaries.
## Logs
To plot the logs of your experiments, you can use the logging utilities provided with the repository.
The trained model folders contain the following experimental artifacts:
```text
checkpoint.pth
eval_metrics.png
evaluation_metrics.csv
losses.csv
training_losses.png
training_metrics.png
variant.yml
```
The `checkpoint.pth` file contains the trained model weights, while `variant.yml` stores the model and experiment configuration. The CSV files contain the corresponding evaluation and training metrics, and the PNG files provide visualizations of the training and evaluation results.
## Attention Maps
To visualize attention maps from the Vision Transformer, you can use the attention-map visualization utilities provided by Segmenter.
For example:
```bash
python -m segm.scripts.show_attn_map checkpoint.pth \
    images/im0.jpg output_dir/ \
    --layer-id 0 \
    --x-patch 0 \
    --y-patch 21 \
    --enc
```
Different options are provided to select the generated attention maps:
- `--enc` or `--dec`: Select encoder or decoder attention maps respectively.
- `--patch` or `--cls`: `--patch` generates attention maps for the patch with coordinates `(x_patch, y_patch)`. `--cls` combined with `--enc` generates attention maps for the CLS token of the encoder. `--cls` combined with `--dec` generates maps for each class embedding of the decoder.
- `--x-patch` and `--y-patch`: Coordinates of the patch to draw attention maps from. These flags are ignored when `--cls` is used.
- `--layer-id`: Select the layer for which the attention map is drawn.
For example, to generate attention maps for the decoder class embeddings:
```bash
python -m segm.scripts.show_attn_map checkpoint.pth \
    images/im0.jpg output_dir/ \
    --layer-id 0 \
    --dec \
    --cls
```
## Video Segmentation: Bug Fire Demonstration
To demonstrate a potential real-world application of the proposed BG-CDF-enhanced semantic segmentation framework, we performed qualitative inference on video footage associated with the August 2026 Bug Fire near Reno, Nevada. The fire affected the HDRFS RRR2 research site, highlighting the importance of remote sensing and image-analysis tools for environmental monitoring and post-fire assessment.
The demonstration uses a trained Vision Transformer model with a ViT-Tiny-Patch16-384 backbone and a Mask Transformer decoder. The checkpoint used for this demonstration is `old_MODEL_FILE_0.7/checkpoint.pth`.
### Video Output
The processed video highlights the model's predicted fire regions in red. The model uses the following four-class taxonomy:
| Class ID | Semantic Class | Visualization Color |
| --- | --- | --- |
| 0 | Background | As defined by the visualization |
| 1 | Fire | Red |
| 2 | Burned area | Orange |
| 3 | Water | Blue |
The generated video is located at `segm/ring_vid.mp4`.
[View the Bug Fire video demonstration](segm/ring_vid.mp4)
### Reproducing the Demonstration
1. Clone this repository and install its dependencies.
2. Download the trained checkpoint from the [Model Zoo](https://drive.google.com/drive/u/1/folders/1DmsfGvBoOKg93RGwIr7kCFrx3FPbMFir).
3. Place the checkpoint in the expected model directory.
4. Provide the Bug Fire video as input to the project's video inference script.
5. Configure the output path and run inference using the same preprocessing and model configuration used for the demonstration.
The checkpoint used for the demonstration was stored at `old_MODEL_FILE_0.7/checkpoint.pth`. Refer to the video inference implementation included in the repository for the exact command-line arguments and processing requirements.
**Important limitation:** This video is a qualitative visualization of model predictions on wildfire footage. It is not a quantitative evaluation of real-time wildfire detection, and the predicted regions should not be interpreted as verified fire boundaries or operational emergency alerts. Additional validation on representative wildfire videos is required before considering such applications.
## BibTeX
If you use the original Segmenter framework, please cite:
```bibtex
@article{strudel2021,
    title={Segmenter: Transformer for Semantic Segmentation},
    author={Strudel, Robin and Garcia, Ricardo and Laptev, Ivan and Schmid, Cordelia},
    journal={arXiv preprint arXiv:2105.05633},
    year={2021}
}
```
## Acknowledgements
The Vision Transformer code is based on the [timm](https://github.com/huggingface/pytorch-image-models) library, and the semantic segmentation training and evaluation pipeline is based on the [Segmenter](https://github.com/rstrudel/segmenter) framework.
The semi-supervised training framework and BG-CDF loss implementation were developed for this project to investigate boundary-aware regularization for semi-supervised semantic segmentation of wildfire imagery.
The Bug Fire video demonstration illustrates a potential application of the framework to wildfire imagery and environmental monitoring. The demonstration is qualitative and should not be interpreted as evidence of operational wildfire-detection performance.
