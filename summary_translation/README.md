# Does Summary Evaluation Survive Translation to Other Languages?

Accepted to appear at the NAACL 2022 Main Conference. Currently available on [Arxiv](https://arxiv.org/abs/2109.08129)


## Data
Most of the data used in this paper can be found in the `data.zip` file. Unzip the data file to a `data` directory to ensure the current code can run (or update the directory in `utils.py`). The paper relies heavily on the `SummEval` dataset, which can be downloaded and explored from its own [repository](https://github.com/Yale-LILY/SummEval). Please see the `readme.txt` in the `data` directory for more information.

## Analysis
The notebooks in this directory can be used to reproduce the figures and statistics of the paper. Due to the large size of the bootstrapped datasets, they must be recreated to produce figures 3 and 5. The notebook `Generate TOST Margin Data.ipynb` should be run before trying to reproduce figures 3 and 4.