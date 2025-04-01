# Instructions

## 1. Data Preparation
The following datasets are required for the project:

- **[MVTEC AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)**  
- **[VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)**  
- **[Real-IAD](https://realiad4ad.github.io/Real-IAD/)**  

For each dataset:
- Place the `data_json/[dataset]/meta.json` file into the corresponding dataset's root directory.  
- For more details, refer to: [Additional Information](https://github.com/zhangzjn/ADer/blob/main/data/README.md).

## 2. Prepare Environment 
Set up the Conda environment using the `requirements.yml` file.

## 3. Configure Dataset Path
Specify the dataset paths by setting `self.data.root` in the file:  
`configs/omniad/dataset_configs.py`.

## 4. Training and Inference
Run the following command for training and inference. You can specify additional settings in `run.py`:  
```bash
python run.py
```
## More Details
Please refer to [ADer](https://github.com/zhangzjn/ADer) 
## Acknowledgements
We would like to express our gratitude to the outstanding works of [MambaAD](https://github.com/lewandofskee/MambaAD) and [ADer](https://github.com/zhangzjn/ADer), among others, for their significant contributions and support to our project.