# BabyBear

Implementation code of ["BabyBear: Cheap inference triage for expensive language models"](https://arxiv.org/abs/2205.11747).


## Get Started

You can generate the results of the paper on emotion recognition with the notebook in Tutorials. This notebook is based on the code available in `run_code/load_emotion.py` and `run_code/emotion.py`. Please use the code available in `run_code` directory for other datasets.

## Source Code

The source code is available at `src/` and is separated into following parts:
* `nlx_babybear.py`: Defines the babybear model
* `inference_triage.py`: Defines the papabear model and the inference triage scheme
* `util_funcs.py`: It contains functions to apply the data preprocessing.

## Contact 

If you have any problems, raise a issue or contact authors [Leila Khalili](leila.khalili@primer.ai), [Yao You](yao.you@primer.ai) or [John Bohannon](john@primer.ai).

## Citation

If you find this repo helpful, we'd appreciate it a lot if you can cite the corresponding paper:
```
@article{khalili2022babybear,
  title={BabyBear: Cheap inference triage for expensive language models},
  author={Khalili, Leila and You, Yao and Bohannon, John},
  journal={arXiv preprint arXiv:2205.11747},
  year={2022},
  url="https://arxiv.org/abs/2205.11747"
}
```
