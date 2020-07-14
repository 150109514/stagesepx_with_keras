import get_data
import time
import train_model
import classify_with_model
import sys
import public_fun


# 用户点击图标完成
# 出现抖音两个字
# 抖音两个字消失
# 进入广告
# 广告结束
# 进入网络缓冲(如果没有缓冲，则广告结束，立刻展示视频标题)
# 展示视频标题
if __name__ == '__main__':
    # 当前日期
    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))

    # 切割视频参数。 视频压缩率：compress_rate, 门限值：threshold, 偏移量：offset, block , date
    '''
    # 全量
    param_list = [[0.2, 0.97, 1, 6, date], [0.2, 0.97, 2, 6, date], [0.2, 0.97, 3, 6, date],
                  [0.2, 0.96, 1, 8, date], [0.2, 0.96, 2, 8, date], [0.2, 0.96, 3, 8, date],
                  [0.2, 0.96, 1, 6, date], [0.2, 0.96, 2, 6, date], [0.2, 0.96, 3, 6, date],
                  [0.2, 0.97, 1, 8, date], [0.2, 0.97, 2, 8, date], [0.2, 0.97, 3, 8, date],
                  [0.2, 0.97, 1, 4, date], [0.2, 0.97, 4, 8, date], [0.2, 0.96, 1, 4, date],
                  [0.2, 0.96, 4, 8, date]
                  ]
    '''
    param_list = [[0.2, 0.97, 0, 8, date]]

    # 训练模型的视频路径
    video_path_for_train = '../videos/train'

    # 训练模型
    picture_path_for_train = '../picture_for_train'
    # model_file_name = '../model/model_ZXGLB_2_GGXQ.h5'
    # model_file_name = '../model/model_FST_2_WR.h5'
    # model_file_name = '../model/model_launch_app.h5'
    # model_file_name = '../model/model_WR_Rotate_Vertical.h5'
    model_file_name = '../model/test.h5'

    # 待预测的视频文件
    # video_path_for_forecast = '../videos/forecast_ZXGLB_2_GGXQ_all'
    # video_path_for_forecast = '../videos/forecast_FST_2_WR_all'
    # video_path_for_forecast = '../videos/forecast_launch_App_all'
    # video_path_for_forecast = '../videos/forecast_WR_Rotate_Vertical'
    video_path_for_forecast = '../videos/test_forecast'

    # 实际结果的csv
    actual_result_csv = video_path_for_forecast + '/actual_result.csv'

    # 视频预测
    cl = None

    # 打印参数
    print('训练视频 = %s' % video_path_for_train)
    # print('用于训练模型的多帧图像 = %s' % picture_path_for_train)
    print('模型文件 = %s' % model_file_name)
    print('预测视频 = %s' % video_path_for_forecast)
    print('---------------------------------------------')

    # 获取键盘输入
    input_list = public_fun.get_keyboard_input()

    # 切割视频
    if input_list[0] == '1\n':
        print("制作训练材料")

        param = param_list[0]
        # 切割出来的视频存放地址
        picture_path_temp = 'cr_' + str(param[0]) + '_th_' + str(param[1]) + '_os_' + str(param[2]) + \
                            '_block_' + str(param[3])
        from_movie_2_picture = '../picture/' + picture_path_temp

        file_name_list = public_fun.get_mp4file_name(video_path_for_train)
        for i in file_name_list:
            get_data.get_range('train', i, param, from_movie_2_picture)
        sys.exit()
    else:
        print("不制作训练材料")

    if input_list[1] == '1\n':
        print("训练SVM模型")
        train_model.train_model_SVM(picture_path_for_train, model_file_name)
    elif input_list[1] == '2\n':
        print("训练Keras模型")
        train_model.train_model_Keras(picture_path_for_train, model_file_name)
    else:
        print("不训练模型")

    # 预测结果
    if input_list[2] == '1\n' or input_list[2] == '2\n':
        SVM_or_Keras = input_list[2]
        for param in param_list:
            # 切割出来的视频存放地址
            picture_path_temp = 'cr_' + str(param[0]) + '_th_' + str(param[1]) + '_os_' + str(param[2]) + \
                                '_block_' + str(param[3])
            from_movie_2_picture = '../picture/' + picture_path_temp

            forecast_result = classify_with_model.calculate_result(cl, SVM_or_Keras, param, from_movie_2_picture,
                                                                   model_file_name, video_path_for_forecast)
            # 把结果写本地
            # print("实际结果为", forecast_result)

            # 对比预期和实际结果，输出结果到csv
            csv_output = from_movie_2_picture + '/output_' + picture_path_temp + '.csv'
            # csv_output = '../final_result.csv'
            public_fun.process_csv(actual_result_csv, forecast_result, csv_output)

    else:
        print("不进行预测")


