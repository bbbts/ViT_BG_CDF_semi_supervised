# Vision Transformer-Based Semi-Supervised Semantic Segmentation with BG-CDF

This repository contains a Vision Transformer (ViT)-based semi-supervised semantic segmentation framework integrated with the Boundary-Guided Class Distribution Flow (BG-CDF) loss. The implementation is based on the [Segmenter](https://github.com/rstrudel/segmenter) framework and includes experiments on the Meta wildfire imagery dataset.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
pip install .
```

## Model Zoo

We provide semi-supervised segmentation models trained on the Meta dataset using a Vision Transformer Tiny backbone (`vit_tiny_patch16_384`), a Mask Transformer decoder, and the BG-CDF loss.

### Meta Dataset

The Meta dataset uses a unified four-class taxonomy consisting of background, fire, burned area, and water.

| Model | Labeled Data | Unlabeled Data | Backbone | Decoder | Download |
| --- | --- | --- | --- | --- | --- |
| MODEL_FILE_0.4 | 40% | 60% | ViT-Tiny-Patch16-384 | Mask Transformer | [Model files](https://drive.google.com/drive/u/1/folders/1DmsfGvBoOKg93RGwIr7kCFrx3FPbMFir) |
| MODEL_FILE_0.5 | 50% | 50% | ViT-Tiny-Patch16-384 | Mask Transformer | [Model files](https://drive.google.com/drive/u/1/folders/1DmsfGvBoOKg93RGwIr7kCFrx3FPbMFir) |
| MODEL_FILE_0.6 | 60% | 40% | ViT-Tiny-Patch16-384 | Mask Transformer | [Model files](https://drive.google.com/drive/u/1/folders/1DmsfGvBoOKg93RGwIr7kCFrx3FPbMFir) |
| MODEL_FILE_0.7 | 70% | 30% | ViT-Tiny-Patch16-384 | Mask Transformer | [Model files](https://drive.google.com/drive/u/1/folders/1DmsfGvBoOKg93RGwIr7kCFrx3FPbMFir) |

Each model folder contains a trained checkpoint, configuration, evaluation metrics, loss logs, and plots.

### Meta Dataset Download

[Download the Meta dataset](https://drive.google.com/drive/u/1/folders/1cDXOGkIqONxV0w9yTdpo6sUfOpoPRJjZ)

## Inference

Download a trained checkpoint and use `inference.py` to generate segmentation predictions for test images. Provide `--gt-dir` to evaluate the predictions against ground-truth masks.

### Example: Inference Using the 40% Labeled-Data Model

Run the following command from the `segm` directory, replacing `/path/to/` with the appropriate base directory on your system:

```bash
python3 inference.py \
    --model-path /path/to/segmenter_SEMI_META_BG_CDF/segm/MODEL_FILE_0.4/checkpoint.pth \
    --input-dir /path/to/Datasets/Meta/images/test/ \
    --output-dir /path/to/segmenter_SEMI_META_BG_CDF/segm/PREDICTION_0.4_new/meta/ \
    --gt-dir /path/to/Datasets/Meta/masks/test/
```

Available labeled-data configurations:

- `MODEL_FILE_0.4`: 40% labeled data and 60% unlabeled data.
- `MODEL_FILE_0.5`: 50% labeled data and 50% unlabeled data.
- `MODEL_FILE_0.6`: 60% labeled data and 40% unlabeled data.
- `MODEL_FILE_0.7`: 70% labeled data and 30% unlabeled data.

Update `--model-path` and `--output-dir` to match the checkpoint and output directory for the model you want to use.

The `--gt-dir` argument can be omitted when ground-truth masks are unavailable and you only want to generate predictions.

## Training

### Meta Dataset Training

The following example trains the model using 40% labeled data and 60% unlabeled data:

```bash
CUDA_VISIBLE_DEVICES=6,7 python3 train.py \
    --dataset-dir /path/to/Datasets/Meta/ \
    --teacher-dir /path/to/segmenter_supervised_META/segm/MODEL_FILE/ \
    --log-dir /path/to/segmenter_SEMI_META_BG_CDF/segm/MODEL_FILE_0.4/ \
    --dataset meta \
    --backbone vit_tiny_patch16_384 \
    --decoder mask_transformer \
    --batch-size 4 \
    --epochs 50 \
    --learning-rate 0.001 \
    --labeled-ratio 0.4 \
    --eval-freq 1
```

For the 70% labeled-data configuration, update the output directory and labeled ratio:

```bash
CUDA_VISIBLE_DEVICES=6,7 python3 train.py \
    --dataset-dir /path/to/Datasets/Meta/ \
    --teacher-dir /path/to/segmenter_supervised_META/segm/MODEL_FILE/ \
    --log-dir /path/to/segmenter_SEMI_META_BG_CDF/segm/MODEL_FILE_0.7/ \
    --dataset meta \
    --backbone vit_tiny_patch16_384 \
    --decoder mask_transformer \
    --batch-size 4 \
    --epochs 50 \
    --learning-rate 0.001 \
    --labeled-ratio 0.7 \
    --eval-freq 1
```

Replace all `/path/to/` placeholders with the actual paths on your system. Ensure that the teacher checkpoint and dataset directories exist before starting training.

Training uses supervised loss on labeled samples and semi-supervised consistency learning on unlabeled samples. BG-CDF adds boundary-aware regularization by comparing teacher and student class-probability profiles sampled across teacher-detected semantic boundaries.

## Logs and Checkpoints

Model folders contain files such as:

```text
checkpoint.pth
eval_metrics.png
evaluation_metrics.csv
losses.csv
training_losses.png
training_metrics.png
variant.yml
```

- `checkpoint.pth`: trained model checkpoint.
- `variant.yml`: model and experiment configuration.
- `evaluation_metrics.csv`: evaluation metrics.
- `losses.csv`: recorded training losses.
- `eval_metrics.png`: evaluation metric plots.
- `training_losses.png`: training loss visualization.
- `training_metrics.png`: training metric visualization.

## Attention Maps

Use the following command to visualize encoder attention maps:

```bash
python -m segm.scripts.show_attn_map checkpoint.pth \
    images/im0.jpg output_dir/ \
    --layer-id 0 \
    --x-patch 0 \
    --y-patch 21 \
    --enc
```

Options:

- `--enc` or `--dec`: selects encoder or decoder attention.
- `--patch` or `--cls`: selects patch or class-token attention maps.
- `--x-patch` and `--y-patch`: specify patch coordinates.
- `--layer-id`: selects the layer to visualize.

To visualize decoder class embeddings:

```bash
python -m segm.scripts.show_attn_map checkpoint.pth \
    images/im0.jpg output_dir/ \
    --layer-id 0 \
    --dec \
    --cls
```

## Video Segmentation: Bug Fire Demonstration

This repository includes a qualitative video-inference demonstration using footage associated with the August 2026 Bug Fire near Reno, Nevada. The fire affected the HDRFS RRR2 research site, illustrating the potential relevance of remote-sensing and image-analysis methods for environmental monitoring and post-fire assessment.

The demonstration uses a Vision Transformer Tiny backbone (`vit_tiny_patch16_384`), a Mask Transformer decoder, and the trained checkpoint located at `old_MODEL_FILE_0.7/checkpoint.pth`.

### Video Output

The processed video highlights predicted fire regions in red. This demonstration focuses on fire predictions only. 

When the command completes successfully, the output video is saved to the location specified by the `--output` argument.

### Running Video Inference

Place the source video and trained checkpoint in their respective locations. Run `video_infer.py` from the `segm` directory.

The following command processes the final 120 seconds of the Bug Fire video using CUDA. The 'seconds' can be adjusted according to need:

```bash
python3 video_infer.py \
    --checkpoint /path/to/checkpoint.pth \
    --source /path/to/input_video.mp4 \
    --output /path/to/save/output_video.mp4 \
    --seconds 120 \
    --device cuda
```

Before running the command:

1. Replace each `/path/to/` placeholder with the correct base directory on your system.
2. Confirm that the checkpoint file exists at the specified `--checkpoint` path.
3. Confirm that the source video exists at the specified `--source` path.
4. Ensure that the CUDA-enabled PyTorch environment is active and CUDA is available.
5. Confirm that the output directory exists and is writable.

The `--seconds 120` argument specifies that the final two minutes of the source video should be processed. The `--device cuda` argument requests GPU inference.

### Limitations

This video is a qualitative visualization, not a quantitative evaluation of real-time wildfire detection. The highlighted regions are model predictions and should not be interpreted as verified fire boundaries, official fire perimeters, or operational emergency alerts. Additional validation on representative wildfire videos is necessary before considering practical deployment.

## BibTeX

```bibtex
@article{strudel2021,
    title={Segmenter: Transformer for Semantic Segmentation},
    author={Strudel, Robin and Garcia, Ricardo and Laptev, Ivan and Schmid, Cordelia},
    journal={arXiv preprint arXiv:2105.05633},
    year={2021}
}
```

## Acknowledgements

The Vision Transformer implementation builds on [timm](https://github.com/huggingface/pytorch-image-models), and the semantic-segmentation pipeline is based on [Segmenter](https://github.com/rstrudel/segmenter).

The semi-supervised training framework and BG-CDF loss implementation were developed to investigate boundary-aware regularization for semi-supervised semantic segmentation of wildfire imagery. The Bug Fire video demonstration illustrates a potential application to wildfire imagery and environmental monitoring; it is qualitative and does not establish operational wildfire-detection performance.
