# CUT_24
Pytorch implementation for SIGIR 24 paper [Aiming at the Target: Filter Collaborative Information for Cross-Domain Recommendation](https://dl.acm.org/doi/10.1145/3626772.3657713)
```
python run_cut.py
```

# Datasets
[Amazon](https://recbole.s3-accelerate.amazonaws.com/CrossDomain/Amazon.zip), [Douban](https://recbole.s3-accelerate.amazonaws.com/CrossDomain/Douban.zip) datasets processed from [RecBole-CDR](https://github.com/RUCAIBox/RecBole-CDR)

For Example, place the [Amazon](https://recbole.s3-accelerate.amazonaws.com/CrossDomain/Amazon.zip) Cloth dataset in the following directory: 

```
dataset/Amazon/AmazonCloth/AmazonCloth.inter
dataset/Amazon/AmazonCloth/AmazonCloth.item
```

# Requirements

```
numpy = '1.21.6'
python = '3.7.12'
recbole = '1.1.1'
torch = '1.13.1'
```

# Acknowledgement

The implementation is based on the open-source recommendation library [RecBole-CDR](https://github.com/RUCAIBox/RecBole-CDR).
