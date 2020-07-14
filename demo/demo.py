from stagesepx.classifier import SVMClassifier
from stagesepx.cutter import VideoCutter
from stagesepx.hook import IgnoreHook
from stagesepx.reporter import Reporter
from stagesepx.video import VideoObject
import pprint
from stagesepx.classifier.keras import KerasClassifier


# 将视频切分成帧
file_name = './video_for_train.mp4'
video = VideoObject(file_name)
# 新建帧，计算视频总共有多少帧，每帧多少ms
video.load_frames()
# 压缩视频
cutter = VideoCutter()
# 计算每一帧视频的每一个block的ssim和psnr。
res = cutter.cut(video, block=6)
# 计算出哪些区间是稳定的，哪些是不稳定的。判断A帧到B帧之间是稳定还是不稳定
stable, unstable = res.get_range(threshold=0.97, offset=2)
# 把分好类的稳定阶段的图片存本地
res.pick_and_save(stable, 100, to_dir='./picture/train_stable_frame', meaningful_name=True)


# 训练模型文件
cl = KerasClassifier(
# 训练轮数
epochs=10
)
cl.train('./train_stable_frame')
cl.save_model('./model.h5', overwrite=True)

# 使用Keras方法进行预测
cl = KerasClassifier()
cl.load_model('./model.h5')

# 将视频切分成帧
file_name = './video_for_forecast.mp4'
# 预加载，大幅度提升分析速度
video = VideoObject(file_name, pre_load=True)
# 新建帧，计算视频总共有多少帧，每帧多少ms
video.load_frames()
# 压缩视频
cutter = VideoCutter()
# 这个hook是干什么的，后续做解释
hook = IgnoreHook(
    size=(0.05, 1),
    overwrite=True,
)
cutter.add_hook(hook)
# 计算每一帧视频的每一个block的ssim和psnr。
res = cutter.cut(video, block=6)
# 计算出哪些区间是稳定的，哪些是不稳定的。判断A帧到B帧之间是稳定还是不稳定
stable, unstable = res.get_range(threshold=0.97, offset=2)
# 把分好类的稳定阶段的图片存本地
res.pick_and_save(stable, 30, to_dir='./picture/forecast_stable', meaningful_name=True)
res.pick_and_save(unstable, 30, to_dir='./picture/forecast_unstable', meaningful_name=True)
# 对切分后的稳定区间，进行归类
classify_result = cl.classify(file_name, stable, keep_data=True)
result_dict = classify_result.to_dict()

# 打印结果
print(result_dict)

# 写html报告
r = Reporter()
r.draw(classify_result, './result.html')
