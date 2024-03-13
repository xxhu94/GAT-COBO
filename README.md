# GAT-COBO

A PyTorch implementation for the paper below:   
**GAT-COBO: Cost-Sensitive Graph Neural Network for Telecom Fraud Detection**.


## Running GAT-COBO
To run the code, you need to have at least Python 3.7 or later versions.  
1.In GAT-COBO/data directory，run`unzip BUPT.zip` and `unzip Sichuan.zip` to unzip the datasets;  
2.Run `python data_process.py` to generate Sichuan and BUPT dataset in DGL;  
3.Run `python main.py` to run GAT-COBO with default settings.  
For other dataset and parameter settings, please refer to the arg parser in train.py. Our model supports both CPU and GPU mode.  

## Repo Structure
The repository is organized as follows:
- `baselines/`:code for all the baselines used in our paper;  
- `data/`: dataset files;  
- `data_process.py`: convert raw node features and adjacency matrix to DGL dataset;
- `main.py`: training and testing GAT-COBO;
- `model.py`: GAT-COBO model implementations;
- `utils.py`: utility functions for EarlyStopping,MixedDropout, MixedLinear, Cost matrix.  


## Running baselines
You can find the baselines in `baselines` directory. For example, you can run Player2Vec using:
```bash
python Player2Vec_main.py 
```

## Citation

```
@article{hu2023gatcobo,
  title={GAT-COBO: Cost-Sensitive Graph Neural Network for Telecom Fraud Detection},
  author={Hu, Xinxin and Chen, Haotian and Zhang, Junjie and Chen, Hongchang and Liu, Shuxin and Li, Xing and Wang, Yahui and Xue, Xiangyang},
  journal={IEEE Transactions on Big Data},
  year={2023},
  doi={10.1109/TBDATA.2024.3352978}
}
```
  
