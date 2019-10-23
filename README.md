# Generating Recognizable Long-term Articulated Objects Motion via Spatio-Temporal Hierarchical Recurrent Network 
This is the pytorch code of my paper Generating Recognizable Long-term Articulated Objects Motion via Spatio-Temporal Hierarchical Recurrent Network.  
Please follow the introduction below to reproduce my results .   

## Required packages

* Anaconda is highly recommend 
* Pytorch 1.2
* Matplotlib 3.0.1
*  FFMPEG 

# Dataset download

* H3.6m dataset

  ```shell
  cd src
  sh ./data/h3.6m/download_h3.6m.sh
  ```

* Mouse dataset

  ```shell
  cd src
  sh ./data/Mouse/download_mouse.sh
  ```

## Reproduce our results

We save our model in the [checkpoint]( https://github.com/p0werHu/human-motion-prediction/tree/master/src/checkpoint ) fold. Our code will search a checkpoint automatically according to your settings. 

```shell
cd src
python train.py --dataset Human --training False --visualize True
```

## Train our network

The main file can be found in [train.py]( https://github.com/p0werHu/human-motion-prediction/blob/master/src/train.py ).

````shell
cd src
python train.py
````

This command will train our network with default settings, i.e. *Human* dataset and *all* actions.

All settings are listed below:

setting | default | values | help
:--:|:--:|:--:|:--:
--gpu|[0]|[.., .., ..]|GPU device ids, list
--training|True|True, False| train or test
--action|all|all, walking, ....|see more in the code
--dataset|Human|Human, Mouse|choose dataset
--visualize|False|True, False|visualize predictions or not, only usable for testing

For more detail configurations, you could refer [config.py]( https://github.com/p0werHu/human-motion-prediction/blob/master/src/config.py )

## Citation

If you find this useful, please cite our work as follows:

```
Not available.
```





