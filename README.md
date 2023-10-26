# Distributed System
Final project for DATA130015.01.

This project implements user-based, item-based, and ALS (Alternating Least Squares) collaborative filtering algorithms for implicit feedback using PySpark. The code files are organized as follows:

* `userBasedCFModel`, `itemBasedCFModel`, and `implicitFeedback` directories contain the implementations of user-based, item-based, and weighted ALS models, respectively.
* In `evaluation.py` and `evaluation_for_als.py`, accuracy is computed for item-based and user-based collaborative filtering as well as weighted ALS.

## Usage

To make recommendations for a specific user:

* User-Based CF

Assuming you want to recommend for a user with the index `USER_INDEX`:

```
cd ./userBasedCFModel
python3 userBasedCFModel.py --user USER_INDEX
```

* Item-Based CF

Assuming you want to recommend for a user with the index `USER_INDEX`:

```
cd ./itemBasedCFModel
python3 itemBasedCFModel.py --user USER_INDEX
```

* Weighted ALS

```
cd ./implicitFeedback
python3 ALSModel.py --user USER_INDEX
```

* Calculate Accuracy

```
python3 evaluation.py
```

* Adjust the Number of Partitions (Using the user-based CF model as an example)

```
cd ./userBasedCFModel
python3 userBasedCFModel.py --user USER_INDEX --parr PARTITION
```
