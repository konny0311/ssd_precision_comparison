## Comparsion of precision with the different volume of training data on Single-Shot MultiBox Detector
---
### Result
### Comparison in all classes
![volume_comparison_total](https://user-images.githubusercontent.com/34429021/60008958-f677cb80-96af-11e9-9c43-29853081bc41.png)
### Comparison in 5 classes
![volume_comparison_5classes](https://user-images.githubusercontent.com/34429021/60008971-fd9ed980-96af-11e9-91c0-15f3eeb8ba7b.png)


### Data

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```

```
root/
  ┣ this dir/
  ┣ data/
      ┣VOCdevkit
        ┣VOC2007
          ┣ImageSets 
            ┣Main # contains txt files of image file names
          ┣JPEGImages
          ┣Annotations # contains xml files of anntation 
        ┣VOC2012
          ┣ImageSets
            ┣Main # contains txt files of image file names
          ┣JPEGImages
          ┣Annotations # contains xml files of anntation 
```

### How to train with reduced data
1. create train/val txt files using create_limmited.py.
2. edit ssd300_training.py. 
3. `python ssd300_training.py`

### How to evaluate with reduced data
1. edit ssd300_evaluate.py. Model weight path and test data path should be specified.
2. `python ssd300_evalute.py`

### Note
This source code is based on https://github.com/pierluigiferrari/ssd_keras

