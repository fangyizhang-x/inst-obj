Instrumented Objects for Assessing Robotic Grasping
====
Implementations for contact estimation via a data-driven method (fully-connected networks). Please refer to the paper (https://arxiv.org/abs/2312.14466) for more details.

## Requirements

  * PyTorch 2
  * Python 3

## Usage

```python evaluate.py --config_file configs/internal_shell_all_probe_denser_no_unseen_face1.yaml```

## Config files
- General training and testing: 
  - Face1: internal_shell_all_probe_denser_no_unseen_face1.yaml
  - Face2: internal_shell_all_probe_denser_no_unseen_face2.yaml
  - Face3: internal_shell_all_probe_denser_no_unseen_face3.yaml
  - Face4: internal_shell_all_probe_denser_no_unseen_face4.yaml
  - Face5: internal_shell_all_probe_denser_no_unseen_face5.yaml
- Real tests: panda_gripper_test_new_zero_ref_all.yaml

## Cite

Please cite our paper if you use this code in your own work:

```
@article{knopke2023towards,
  title={Towards Assessing Compliant Robotic Grasping from First-Object Perspective via Instrumented Objects},
  author={Knopke, Maceon and Zhu, Liguo and Corke, Peter and Zhang, Fangyi},
  journal={arXiv preprint arXiv:2312.14466},
  year={2023}
}
```
