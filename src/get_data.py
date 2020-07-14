from stagesepx.cutter import VideoCutter
from stagesepx.hook import IgnoreHook, CropHook, BaseHook, ExampleHook, FrameSaveHook
from stagesepx.video import VideoObject
import re


def get_range(_train_or_forecast, _forecast_file_name, _param, _picture_path):
    file_name = re.search(r'\\(.*).mp4', str(_forecast_file_name), re.M | re.I).group(1)
    video = VideoObject(_forecast_file_name, pre_load=True)
    # 新建帧，计算视频总共有多少帧，每帧多少ms
    video.load_frames()
    # 直接切割视频
    # 压缩视频
    cutter = VideoCutter(compress_rate=_param[0])

    # 添加Hook
    '''
    convert_size_and_offset:1zai 51 - origin size: ((468, 216)) 显示的是整张图的尺寸，坐标原点在左上角。
    468是高，216是宽。(468, 216)是整个画面的右下角
    stagesepx.hook:convert_size_and_offset:153 - size: (23.4, 216) 显示的是被屏蔽或者是被选择的区域的高和宽。高是23.4，宽是216。
    convert_size_and_offset:160。final range h: (0, 23), w: (0, 216) 显示的是被屏蔽或者是被选择的区域。
    从左上角开始计算，屏蔽掉0到23高度，整个宽度的区域。
    
    如果要统计竖屏切换到横屏场景的耗时。为了防止录屏文件中，横屏后的"五日分数图"内容被压缩的太小，建议在录屏时选择横屏录制。
    横屏录制场景下，某一帧图像中，App的内容处于竖屏状态下时，该帧图像表现为，左边一部分黑屏，中间是手机竖屏页面，右边还是一部分黑屏。
    此时如果要屏蔽或者选择某一块区域，要把黑色区域的尺寸也算在内。
    如果只通过中间部分有手机内容图像的尺寸，来选择或者屏蔽部分区域，则会出现被选择的区域和预期不符的情况。
    
    # 屏蔽盘口数据 size=(0.1, 1),
    # 屏蔽手机导航栏 size=(0.05, 1),
    '''

    '''
    # 屏蔽帧的右边
    hook_ignore1 = IgnoreHook(
        size=(1, 1),
        offset=(0, 0.55),
        overwrite=True,
    )
    # 屏蔽帧的左边
    hook_ignore2 = IgnoreHook(
        size=(1, 0.45),
        overwrite=True,
    )
    # 屏蔽帧的上边
    hook_ignore3 = IgnoreHook(
        size=(0.35, 1),
        overwrite=True,
    )
    # 屏蔽帧的下边
    hook_ignore4 = IgnoreHook(
        size=(1, 1),
        offset=(0.6, 0),
        overwrite=True,
    )
    hook_crop = CropHook(
        size=(0.2, 0.2),
        overwrite=True,
    )
    '''

    # cutter.add_hook(hook_ignore1)
    # cutter.add_hook(hook_ignore2)
    # cutter.add_hook(hook_ignore3)
    # cutter.add_hook(hook_ignore4)
    # cutter.add_hook(hook_crop)
    # hook_save_frame = FrameSaveHook('../frame_save_dir')
    # cutter.add_hook(hook_save_frame)

    # 计算每一帧视频的每一个block的ssim和psnr。block=4则算16个part的得分
    # res = cutter.cut(video)
    res = cutter.cut(video, block=_param[3])
    # 计算出哪些区间是稳定的，哪些是不稳定的。判断A帧到B帧之间是稳定还是不稳定
    # 是不是在这里就决定，把稳定的A到B帧，放到一个文件夹。如果offset大，就扩大A到B的间隔。
    stable, unstable = res.get_range(threshold=_param[1], offset=_param[2])
    # stable, unstable = res.get_range(threshold=_param[1], offset=_param[2], limit=5,)
    # stable, unstable = res.get_range(threshold=_param[1], offset=_param[2], psnr_threshold=0.85)
    # stable, unstable = res.get_range(threshold=_param[1])
    if _train_or_forecast == 'train':
        print("pick_and_save")
        # 把分好类的稳定阶段的图片存本地
        res.pick_and_save(stable, 20, to_dir=_picture_path + '/train_stable_' + file_name, meaningful_name=True)
        # 把分好类的不稳定阶段的图片存本地
        res.pick_and_save(unstable, 40, to_dir=_picture_path + '/train_unstable_' + file_name,
                           meaningful_name=True)
    else:
        res.pick_and_save(stable, 20, to_dir=_picture_path + '/forecast_stable_' + file_name, meaningful_name=True)
        # 把分好类的不稳定阶段的图片存本地
        res.pick_and_save(unstable, 40, to_dir=_picture_path + '/forecast_unstable_' + file_name,
                           meaningful_name=True)

    return stable





