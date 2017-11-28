# FMG
The code for our KDD17 paper "Meta-Graph Based Recommendation Fusion over Heterogeneous Information Networks"

Readers are welcomed to fork this repository to reproduce the experiments and follow our work. Just remmeber to cite our paper

@inproceedings{zhao2017meta,
  title={Meta-Graph Based Recommendation Fusion over Heterogeneous Information Networks},
  author={Zhao, Huan and Yao, Quanming and Li, Jianda and Song, Yangqiu and Lee, Dik Lun},
  booktitle={KDD},
  pages={635--644},
  year={2017}
}

The Yelp-50K and Amazon-50K are released first: https://www.dropbox.com/s/io915s8mdwph46e/data.zip?dl=0

Extract the files and keep the structure of the directory as it. 

A typical example is run as: 
    
    python run_exp.py config/yelp-50k.yaml -reg 0.5

One may read the comment in files in directory config for more information.

To run the code sucessfully, you also need to 
  
    1. Put "data" in the project directory.
    2. Create directory "log" in the project by the command "mkdir log".
    3. Create directory "fm_res" in the project by the command "mkdir fm_res".
