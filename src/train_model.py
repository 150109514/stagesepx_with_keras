from stagesepx.classifier import SVMClassifier
from stagesepx.classifier.keras import KerasClassifier


def train_model_SVM(_train_picture_path, _model_file_name):
    cl = SVMClassifier(
        # 默认情况下使用 HoG 进行特征提取
        # 你可以将其关闭从而直接对原始图片进行训练与测试：feature_type='raw'
        feature_type="hog",
        # 默认为0.2，即将图片缩放为0.2倍
        # 主要为了提高计算效率
        # 如果你担心影响分析效果，可以将其提高
        compress_rate=0.2,
        # 或者直接指定尺寸
        # 当压缩率与指定尺寸同时传入时，优先以指定尺寸为准
        # target_size=(200, 400),
    )

    # 加载待训练数据
    cl.load(_train_picture_path)
    # 在加载数据完成之后需要先训练
    cl.train()
    cl.save_model(_model_file_name, overwrite=True)

    return cl


def train_model_Keras(_train_picture_path, _model_file_name):
    cl = KerasClassifier(
        # 轮数
        epochs=10,
        compress_rate=0.2,
        # 保证数据集的分辨率统一性
        # target_size=(600, 800),
    )
    cl.train(_train_picture_path)
    cl.save_model(_model_file_name, overwrite=True)

    return cl
