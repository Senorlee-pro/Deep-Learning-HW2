# Deep-Learning-HW2
## Environment setup
You may want to create a new `conda` environment for the project.
```bash
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

## Train
For training MLP and CNN baseline models, run `train_mlp.py` and `train_cnn.py`.

## Test
For testing your models, run `test_model.py`.

Note that you should correctly set `MODEL_PATH` and `MODEL_TYPE` before starting the test.

## Robustness and error analysis
Check `robustness_analysis.py` and `error_analysis.py` for reference.
