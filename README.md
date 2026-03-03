# FID, sFID, IS, Precision & Recall evaluation in PyTorch

This is a PyTorch implementation of [Fréchet Inception Distance](https://arxiv.org/abs/1706.08500) (FID), [spatial Fréchet Inception Distance](https://arxiv.org/abs/2103.03841) (sFID), [Inception Score](https://arxiv.org/abs/1606.03498) (IS), [Precision & Recall](https://arxiv.org/abs/1904.06991) evalution for image generation task.

> A widely acknowledged TensorFlow implementation is [guided-diffusion/evaluations](https://github.com/openai/guided-diffusion/tree/main/evaluations)

## Usage

1. Download [InceptionV3 wights](https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth) provided by [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
2. Prepare reference statistics:
    - for ImageNet, you can download reference npz file (`ref_npz`) from [guided-diffusion/evaluations](https://github.com/openai/guided-diffusion/tree/main/evaluations)
    - for customized datasets, you need to prepare a sets of reference images (`ref_imageset`)
    - (optional) you can compress your `ref_imageset` into `ref_npz` by

    ```
    python metrics.py --weights path/to/inception_weights \
        --src-path path/to/ref_imageset --save-src-npz-path path/to/ref_npz
    ```
3. Generate source images with your model (`src_imageset`), e.g. 50000 images typically. Then calculate FID, sFID, IS, precision & recall:

```
python metrics.py --weights path/to/inception_weights \
    --src-path path/to/src_imageset --ref-path path/to/ref_npz
# or
python metrics.py --weights path/to/inception_weights \
    --src-path path/to/src_imageset --ref-path path/to/ref_imageset
```

## Acknowledgements

This project is based on following projects:

- [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
- [guided-diffusion](https://github.com/openai/guided-diffusion)
- [TTUR](https://github.com/bioinf-jku/TTUR)

## License

This implementation is licensed under the Apache License 2.0.