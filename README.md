# Rank List Sensitivity of Recommender Systems to Interaction Perturbations (ACM CIKM 2022)

Overview
---------------
**Rank List Sensitivity of Recommender Systems to Interaction Perturbations**  
[Sejoon Oh](https://sejoonoh.github.io/), [Berk Ustun](https://www.berkustun.com/), [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/), and [Srijan Kumar](https://www.cc.gatech.edu/~srijan/)  
*[ACM International Conference on Information and Knowledge Management (CIKM)](https://www.cikm2022.org/), 2022*  

Link to the paper PDF - will be added soon

**CASPER** is a novel framework to measure the model stability of recommender systems against input data perturbations. You can obtain the sensitivity of your recommender systems against the interaction-level perturbation done by CASPER perturbation. CASPER perturbs an interaction (deletion, item replacement, or insertion) that impacts the largest number of interactions when it is perturbed. Since CASPER is a ***model-agnostic*** perturbation (except the maximum sequence length of a model), you can easily find an interaction to perturb by processing the dataset and indicating the perturbation types.

If you make use of this code, paper, or the datasets in your work, please cite the following paper:
```
 @inproceedings{oh2022rank,
	title={Rank List Sensitivity of Recommender Systems to Interaction Perturbations},
	author={Oh, Sejoon and Ustun, Berk and McAuley, Julian and Kumar, Srijan},
	booktitle={Proceedings of the 31st ACM International Conference on Information and Knowledge Management},
	year={2022},
	organization={ACM}
 }
```

Usage
---------------

The detailed execution procedure of **CASPER** is given as follows.

1) Install all required libraries by "pip install -r requirements.txt" (Python 3.6 or higher version is required).
2) "src/main.py" and "src/lstm.py" include data loading/perturbation creation and training/evaluation parts of CASPER on the LSTM model, respectively. 
3) "python src/main.py [arguments]" will execute CASPER on the LSTM model with arguments, and specific information of the arguments are as follows.

````
--data_path: path of an input dataset (e.g., data/wikipedia.tsv)
--epochs: number of epochs for training (default: 50)
--gpu: GPU number will be used for experiments (default: 0)
--output: name of the output log file (e.g., wikipedia_stability)
--num_perturbation: number of input perturbations (default: 1)
--max_seq_len: maximum sequence length for CASPER perturbation (default: 50)
--test_data_ratio: last K% of interactions of each user will be used as test data (default: 0.1)
--batch_size: mini-batch size for training (default: 1024)
--learning_rate: learning rate for training (default: 0.001)
````

Note that **CASPER** is a ***model-agnostic*** perturbation framework, which can be applied to any recommendation models.  
The provided codes are targeting a LSTM model, and you can customize the code to your models.

Demo
---------------
To run the demo, please follow the following procedure. **CASPER** demo will be executed with the Wikipedia dataset.

	1. Check permissions of files (if not, use the command "chmod 777 *")
	2. Execute "./demo.sh"
	3. Check "wikipedia_stability" for the demo result of CASPER on Wikipedia dataset
  
Tested Environment
---------------
We tested our proposed method **CASPER** in Azure Standard-NC24 machines equipped with 4 NVIDIA Tesla K80 GPUs with Intel Xeon E5-2690 v3 processor.
  
Datasets
---------------
The datasets used in the paper are available at [this link](https://drive.google.com/file/d/1YAyI8Yy-xgU6h4xaWNfnBJwVnufMM85m/view?usp=sharing).  
The input data format must be tab-separated, integer-type for user, items, and timestamps (see the data/wikipedia.tsv).
