from stagesepx.classifier import SVMClassifier
from stagesepx.classifier.keras import KerasClassifier
import get_data
import public_fun


def calculate_result(_cl, _SVM_or_Keras, _param, _from_movie_2_picture, _model_file, _video_path_for_forecast):

    if _SVM_or_Keras == '1\n':
        print('使用SVM进行预测')
        cl = SVMClassifier(
            # 默认情况下使用 HoG 进行特征提取。你可以将其关闭从而直接对原始图片进行训练与测试：feature_type='raw'
            feature_type="hog",
            # 默认为0.2，即将图片缩放为0.2倍。主要为了提高计算效率,如果你担心影响分析效果，可以将其提高
            compress_rate=_param[0],
        )
        # 加载待训练数据
        cl.load_model(_model_file)
    elif _SVM_or_Keras == '2\n':
        # 分析视频_聚类
        print('使用KerasClassifier进行预测')
        cl = KerasClassifier(
            compress_rate=_param[0],
            # 在使用时需要保证数据集格式统一（与训练集）。因为 train_model.py 用了 600x800，所以这里设定成一样的
            # target_size=(600, 800),
        )
        cl.load_model(_model_file)

    # 开始预测
    _forecast_result = []

    # 获取forecast文件夹的mp4文件列表
    forecast_video_list = public_fun.get_mp4file_name(_video_path_for_forecast)

    for i in forecast_video_list:
        # 分析视频_切割视频
        stable = get_data.get_range('forecast', i, _param, _from_movie_2_picture)
        classify_result = cl.classify(i, stable, keep_data=True)
        result_dict = classify_result.to_dict()

        _forecast_result.append(public_fun.write_result_to_local(i, _from_movie_2_picture,
                                                                 result_dict, classify_result))
        # _forecast_result = [['5.mp4', '3.161888888888889', '1',
        # '4.155888888888889', '1', '8.040888888888889', '1']]

    return _forecast_result

