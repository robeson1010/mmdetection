
# Rice detection based Faster-Rcnn

## Introduction

A experiment that test Cascade Faster-Rcnn performance on Drone image. The model was based on mmdetection.

mmdetection is an open source object detection toolbox based on PyTorch. It is
a part of the open-mmlab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).


2. Augmentation
=======
To perform evaluation after testing, add `--eval <EVAL_TYPES>`. Supported types are:
`[proposal_fast, proposal, bbox, segm, keypoints]`.
`proposal_fast` denotes evaluating proposal recalls with our own implementation,
others denote evaluating the corresponding metric with the official coco api.
>>>>>>> origin/master

Albu aug library

```Python
        self.light=A.Compose([A.RGBShift(),A.InvertImg(),
            A.Blur(),
            A.GaussNoise(),
            A.Flip(),
            A.RandomRotate90()], bbox_params={'format':'pascal_voc','min_visibility': 0.4, 'label_fields': ['category_id']}, p=1)
```


## Model config
```
dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations.pkl',
        img_prefix=data_root + 'train/',
        img_scale=[(384, 768), (768, 768)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True))
```


<<<<<<< HEAD
## Result
=======
## Train a model

mmdetection implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

### Distributed training (Single or Multiples machines)

mmdetection potentially supports multiple launch methods, e.g., PyTorch’s built-in launch utility, slurm and MPI.
>>>>>>> origin/master

![1](./images/1.png)

![2](./images/2.png)

<<<<<<< HEAD
![3](./images/3.png)

=======
Expected results in WORK_DIR:

- log file
- saved checkpoints (every k epochs, defaults=1)
- a symbol link to the latest checkpoint

**Important**: The default learning rate is for 8 GPUs. If you use less or more than 8 GPUs, you need to set the learning rate proportional to the GPU num. E.g., modify lr to 0.01 for 4 GPUs or 0.04 for 16 GPUs.

### Non-distributed training

Please refer to `tools/train.py` for non-distributed training, which is not recommended
and left for debugging. Even on a single machine, distributed training is preferred.

### Train on custom datasets

We define a simple annotation format.

The annotation of a dataset is a list of dict, each dict corresponds to an image.
There are 3 field `filename` (relative path), `width`, `height` for testing,
and an additional field `ann` for training. `ann` is also a dict containing at least 2 fields:
`bboxes` and `labels`, both of which are numpy arrays. Some datasets may provide
annotations like crowd/difficult/ignored bboxes, we use `bboxes_ignore` and `labels_ignore`
to cover them.

Here is an example.
```
[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        'ann': {
            'bboxes': <np.ndarray> (n, 4),
            'labels': <np.ndarray> (n, ),
            'bboxes_ignore': <np.ndarray> (k, 4),
            'labels_ignore': <np.ndarray> (k, ) (optional field)
        }
    },
    ...
]
```

There are two ways to work with custom datasets.

- online conversion

  You can write a new Dataset class inherited from `CustomDataset`, and overwrite two methods
  `load_annotations(self, ann_file)` and `get_ann_info(self, idx)`, like [CocoDataset](mmdet/datasets/coco.py) and [VOCDataset](mmdet/datasets/voc.py).

- offline conversion

  You can convert the annotation format to the expected format above and save it to
  a pickle or json file, like [pascal_voc.py](tools/convert_datasets/pascal_voc.py).
  Then you can simply use `CustomDataset`.

## Technical details

Some implementation details and project structures are described in the [technical details](TECHNICAL_DETAILS.md).

## Citation

If you use our codebase or models in your research, please cite this project.
We will release a paper or technical report later.

```
@misc{mmdetection2018,
  author =       {Kai Chen and Jiangmiao Pang and Jiaqi Wang and Yu Xiong and Xiaoxiao Li
                  and Shuyang Sun and Wansen Feng and Ziwei Liu and Jianping Shi and
                  Wanli Ouyang and Chen Change Loy and Dahua Lin},
  title =        {mmdetection},
  howpublished = {\url{https://github.com/open-mmlab/mmdetection}},
  year =         {2018}
}
```
>>>>>>> origin/master
