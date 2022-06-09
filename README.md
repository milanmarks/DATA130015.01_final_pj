# distributed_system
Final project for DATA130015.01.

Implement user-based, item-based and ALS for implicit feedback collaborative filtering algorithm by pyspark.

代码文件中，`userBasedCFModel`、`itemBasedCFModel`、`implicitFeedback`文件夹中分别包
含了对基于用户、基于物品、带权重的 ALS 模型的实现。在 `evaluation.py` 和 `evaluation_for_als.py`
中分别对基于物品和用户、带权重的 ALS 的准确度进行计算。

## 用法

为特定用户推荐：

* 基于用户

假设为用户序列`USER_INDEX`推荐：

```
cd ./userBasedCFModel
python3 userBasedCFModel.py --user USER_INDEX
```

* 基于物品

假设为用户序列`USER_INDEX`推荐：

```
cd ./itemBasedCFModel
python3 itemBasedCFModel.py --user USER_INDEX
```

* 带权重的ALS

```
cd ./implicitFeedback
python3 ALSModel.py --user USER_INDEX
```
