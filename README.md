# Cassava Leaf Disease Classification

The repository contains a deeplearning pipeline for identifying Cassava Leaf Disease for a ongoning [Kaggle Competition](https://www.kaggle.com/c/cassava-leaf-disease-classification/overview). 

To run a 5 fold experiment you can run the below shell script.

```
./run_kfold.sh
```

By default the code uses

- resnext50_32x4d_ssl pretrained model from torch hub.
- If you have multiple GPU's, then the code automatically utilizes all of them.
- By default the code takes advantage of Mixed Precision.

A single model achieves 0.883 on LB
A 5-Fold model achieves 0.894 on LB

