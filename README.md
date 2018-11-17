
# Rice detection based Faster-Rcnn

## Introduction

A experiment that test Faster-Rcnn performance on Drone image. The model was based on mmdetection.

mmdetection is an open source object detection toolbox based on PyTorch. It is
a part of the open-mmlab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).


## Data and processing

1. Random Crop as 768* 768  20 times for each image.

```Python
annotations=[]
# namenum=1
for i in tqdm.tqdm(range(len(annotationspd))):
    bbox=annotationspd.iloc[i,0]['bboxes']
    labels=annotationspd.iloc[i,0]['labels']
    xxx=io.imread(os.path.join('/data2/rice2/',annotationspd.iloc[i,1]))
    
    for m in range(20):
        augmented=aug(image=xxx,bboxes=bbox, category_id=labels)
        newbb=augmented['bboxes']
        newbb=np.array(newbb, ndmin=2)
        newlabels=np.ones((newbb.shape[0]))
        filename='%02d'%(i)+'-'+ '%02d'%(m)+'.png'
        oneimg={'filename':filename,
                        'width': int(768),
                        'height': int(768),
                        'ann': {
                            'bboxes': newbb.astype(np.float32),
                            'labels': newlabels.astype(np.int64),
                            'bboxes_ignore':bboxes_ignore.astype(np.float32)}
                       }
        if(oneimg['ann']['bboxes'].shape[1]==4):
            annotations.append(oneimg)
            io.imsave(os.path.join('/data2/rice/train',filename),augmented['image'])
```

2. Augmentation

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


## Result

![1](./images/1.png)

![2](./images/2.png)

![3](./images/3.png)

