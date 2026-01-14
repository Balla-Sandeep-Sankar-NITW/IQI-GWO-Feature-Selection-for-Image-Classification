
**please follow the commands one by one without missing anything .......**


Project :  Image Classification using SVM Optimized by Improved Quantumly Inspired Grew Wolf Optimization

**Download the DataSets from these link :**

LC25000
https://www.kaggle.com/datasets/javaidahmadwani/lc25000/data

Plants Type Datasets
https://www.kaggle.com/datasets/yudhaislamisulistya/plants-type-datasets

IIITDMJ_Smoke
https://data.mendeley.com/datasets/4mn2g8cnsf/1

Save these datasets in "Datsets" Folder

command : 
    pip install numpy joblib scikit-learn

**Extract the datasets :**

command :

python unzip.py --path "Datasets"

NOTE : In the IIITDMJ_Smoke.zip inside another zip file is there , so you have to extract it twice 

command :

python unzip.py --path "DataSets\IIITDMJ_Smoke"

**Data Spliting :**

command :

python split_data.py --path "DataSets/lung_colon_image_set/Train and Validation Set" --splits 2 --names train,val --ratio 800/9:100/8

NOTE : For reamining data sets no need to split the dataset , because the dataset having train , val , test data seperately


**Run main.py for DataSets one by one :**

1. lung_colon_image_set :

python main.py --train "DataSets\lung_colon_image_set\Train and Validation Set\train_data" --val "DataSets\lung_colon_image_set\Train and Validation Set\val_data" --test "DataSets/lung_colon_image_set/Test Set"  

2. split_ttv_dataset_type_of_plants :

python main.py --train "DataSets/split_ttv_dataset_type_of_plants/Train_Set_Folder" --val "DataSets/split_ttv_dataset_type_of_plants/Validation_Set_Folder" --test "DataSets/split_ttv_dataset_type_of_plants/Test_Set_Folder"  

3. IIITDMJ_Smoke :

command :
python main.py --train "DataSets/IIITDMJ_Smoke/IIITDMJ_Smoke/train" --val "DataSets/IIITDMJ_Smoke/IIITDMJ_Smoke/val" --test "DataSets/IIITDMJ_Smoke/IIITDMJ_Smoke/test"

