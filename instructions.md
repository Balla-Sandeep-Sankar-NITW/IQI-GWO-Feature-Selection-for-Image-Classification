# Image Classification using SVM Optimized by Improved Quantumly Inspired Grey Wolf Optimization

Follow the commands below in order without skipping any step.

## Download the Datasets

Download the following datasets:

- **LC25000**  
  https://www.kaggle.com/datasets/javaidahmadwani/lc25000/data

- **Plants Type Datasets**  
  https://www.kaggle.com/datasets/yudhaislamisulistya/plants-type-datasets

- **IIITDMJ_Smoke**  
  https://data.mendeley.com/datasets/4mn2g8cnsf/1

Save all the downloaded datasets in the **`DataSets`** folder.

---

## Install the Required Packages

Run the following command:

```bash
pip install numpy joblib scikit-learn
```

---

## Extract the Datasets

Run the following command:

```bash
python unzip.py --path "DataSets"
```

> **Note:** The `IIITDMJ_Smoke.zip` file contains another ZIP file inside it. Therefore, you need to extract it twice.

Run the following command:

```bash
python unzip.py --path "DataSets\IIITDMJ_Smoke"
```

---

## Split the Data

Run the following command:

```bash
python split_data.py --path "DataSets/lung_colon_image_set/Train and Validation Set" --splits 2 --names train,val --ratio 800/9:100/8
```

> **Note:** The remaining datasets do not need to be split because they already contain separate train, validation, and test folders.

---

## Run `main.py` for Each Dataset

### `lung_colon_image_set`

```bash
python main.py --train "DataSets\lung_colon_image_set\Train and Validation Set\train_data" --val "DataSets\lung_colon_image_set\Train and Validation Set\val_data" --test "DataSets/lung_colon_image_set/Test Set"
```

### `split_ttv_dataset_type_of_plants`

```bash
python main.py --train "DataSets/split_ttv_dataset_type_of_plants/Train_Set_Folder" --val "DataSets/split_ttv_dataset_type_of_plants/Validation_Set_Folder" --test "DataSets/split_ttv_dataset_type_of_plants/Test_Set_Folder"
```

### `IIITDMJ_Smoke`

Run the following command:

```bash
python main.py --train "DataSets/IIITDMJ_Smoke/IIITDMJ_Smoke/train" --val "DataSets/IIITDMJ_Smoke/IIITDMJ_Smoke/val" --test "DataSets/IIITDMJ_Smoke/IIITDMJ_Smoke/test"
```
