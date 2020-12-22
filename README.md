# Capstone-Zhen

This repo is to document the capstone project of Fruit Classification. 

* The model we used is a very simple Convolutional neural network: 3 convolutional layers followed by pooling layers. As a convention, batch normalization and dropout were used to reduce the eisk of overfitting. 

## Dependencies
Prior to running all the code, adding all the following libraries:

* pytorch (1.5.1) and torchvision (0.6.0) 

  (GPU version) 
  ```conda install pytorch torchvision cudatoolkit=10.2 -c pytorch```
  
  (CPU version) 
  ```conda install pytorch torchvision```
  
* glob 
* matplotlib (3.2.2)
* numpy (1.18.1)
* tqdm (4.46.1) (visualize the ongoing process)

  ```conda install glob matplotlib numpy tqdm```

* ##### other libraries will be updated as we go 

## Train 
sh run.sh on linux or any Unix distribution , or directly by
``` 
python main.py train  --epochs [number of training epochs, default is 10 ] \
                      --batch-size [batch size for training, default is 32] \
                      --dataset [path to training dataset, the path should point to a folder] \
                      --save-model-dir [path to folder where trained model will be saved.] \
                      --checkpoint-model-dir [path to folder where checkpoints of trained models will be saved] \
                      --image-size [the size of training images, default is 100 X 100] \
                      --val-rate [the rate of training data used as validation set] \
                      --cuda [enables gpu] \
                      --seed [random seed for training] \
                      --lr [learning rate, default is 1e-3]             
```     

## Evulate
sh run.sh on linux or any Unix distribution , or directly by
``` 
python main.py eval   --dataset [path to training dataset, the path should point to a folder] \
                      --model [path to trained model, should be a exact path] \
                      --cuda [enables gpu] \
                      --batch-size [batch size for training, default is 64, depends on the memory of gpu] \
                               
```     



