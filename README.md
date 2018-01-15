# FMG
The code KDD17 paper "Meta-Graph Based Recommendation Fusion over Heterogeneous Information Networks"

Readers are welcomed to fork this repository to reproduce the experiments and follow our work. Please kindly cite our paper

    @inproceedings{zhao2017meta,
    title={Meta-Graph Based Recommendation Fusion over Heterogeneous Information Networks},
    author={Zhao, Huan and Yao, Quanming and Li, Jianda and Song, Yangqiu and Lee, Dik Lun},
    booktitle={KDD},
    pages={635--644},
    year={2017}
    }

## Instructions

For the sake of ease, a quick instruction is given for readers to reproduce the whole process on yelp-50k dataset. Note that the programs are testd on CentOS release 6.9(Python 2.7.)

### Prerequisites

1. Download the yelp-50k and amazon-50k : https://www.dropbox.com/s/io915s8mdwph46e/data.zip?dl=0
2. Extract data from zip file and put it in the project, then iteratively create directories **"sim\_res/path\_count"** and **"mf\_features/path\_count"** in directory **"data/yelp-50k/exp_split/1/"**.
3. Create directory **"log"** in the project by "mkdir log".
4. Create directory **"fm\_res"** in the project by "mkdir fm\_res".

###Meta-graph Similarity Matrices Computation.
To generate the similarity matrices on yelp-50k dataset, run

	python 200k_commu_mat_computation.py yelp-50k all 1
The arguments are explained in the following:
	
	yelp-50k: specify the dataset.
	all: run for all pre-defined meta-graphs.
	1: run for the split dataset 1, i.e., exp_split/1
### Meta-graph Latent features generation.
To generate the latent features by MF based on the simiarity matrices, run
    
    python mf_features_generator.py yelp-50k all 1

The arguments are the same as the above ones.

Note that, to improve the computation efficiency, some modules are implements with C and called in python(see *load_lib* method in mf.py). Thus to successfully run mf\_features\_generator.py, you need to compile two C source files. The following scripts are tested on CentOS, and readers may take as references.

	gcc -fPIC --shared setVal.c -o setVal.so
	gcc -fPIC --shared partXY.c -o partXY.so

After the compiling, you will get two files in the project directory "setVal.so" and "partXY.so".

### FMG
After obtain the latent features, then the readers can run FMG model as following:
    
    python run_exp.py config/yelp-50k.yaml -reg 0.5

One may read the comment in files in directory config for more information.

## Misc
If you have any questions about this project, you may open issues, thus it can help more people who are interested in this project.
