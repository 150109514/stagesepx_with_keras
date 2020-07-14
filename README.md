# stagesepx_with_keras


# 想做什么

有一天，开发人员说，我在X.X.X版本中，做了XXXXXXX改变，App的页面切换"**应该**"能更快一点、"**应该**"能更快流畅一点。测试人员，怎么验证？

# 怎么做

## 埋点&眼睛看

在App端，统计耗时的主要方法有：

- **埋点**：开发人员喜欢使用的方法。对于开发人员而言，在某项操作的起点、终点各打一个点，做个差，拿到耗时数据。

- **眼睛看**：测试人员更喜欢使用的方法。这种方法统计出来的数据，是用户感知的最真实数据。测试人员也弄不到能打印日志的Debug版本测试包，也不知道应该怎么埋点。所以测试人员更喜欢"眼睛看"。

## 怎么用眼睛"看"

应付工作的，直接看App界面，拿个秒表，某项操作的起点记个点，终点记个点。早上俩小时，就干这一件事。

认真工作的，用手机录个视频，拿个分帧软件，一帧一帧看。这种方法能保证数据准确，但是效率太低。还是早上俩小时，就干这一件事。

# 怎么提升统计分析的效率

## 关于stagesepx

这里给大家介绍一种基于图像处理与机器学习的自动化视频分析方案 —— stagesepx。

stagesepx能将视频拆分为帧，并将其划分为多个阶段。在此之后，你可以清晰地得知视频包含了几个阶段、以及每个阶段在干什么，且这一切都是自动完成的。

## 简单的例子

App启动页性能优化。这几乎是每个App开发团队绕不开的工作之一。下面将通过一个简单的例子给大家介绍如何使用stagesepx做App启动耗时统计。

我们用手机自带的录屏工具，录制了一段"国元点金"App的启动视频。我们可以看到App启动可以分成以下几个阶段：

- A.手机桌面


- B.用户点击App图标


- C.打开App，显示"国元点金"首屏


- D.进入App首页


我们将App启动耗定义为：D - B。App启动的主要阶段如下：![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/launch_dianjin_app.jpg)

## 怎么使用stagesepx进行性能统计

本文介绍的使用stagesepx方式统计分析某项操作的耗时，主要是基于Keras分析方法。至于SVM分析方法，由于准确度没有前者高，固不在此进行详细介绍。使用stagesepx方式进行耗时统计的整个过程可以被分为"切割视频"、"判断稳定区间"、"训练模型"和"预测分析"四个步骤。

### 视频切割和判定稳定区间

	from stagesepx.cutter import VideoCutter
	from stagesepx.video import VideoObject
	
	# 将视频切分成帧
	file_name = './new_Screenrecorder-2020-06-26-06-14-58-29.mp4'
	video = VideoObject(file_name, pre_load=True)
	# 新建帧，计算视频总共有多少帧，每帧多少ms
	video.load_frames()
	# 压缩视频
	cutter = VideoCutter()
	# 计算每一帧视频的每一个block的ssim和psnr。
	res = cutter.cut(video, block=6)
	# 计算出判断A帧到B帧之间是稳定还是不稳定
	stable, unstable = res.get_range(threshold=0.96)
	# 把分好类的稳定阶段的图片存本地
	res.pick_and_save(stable, 20, to_dir='./picture/stable_frame', meaningful_name=True)

通过上述过程，stagesepx可以将一个视频文件切分成若干帧图像，并将其认定的处于稳定状态的图像保存到"stable_frame"文件夹内。

在"stable_frame"文件夹中，帧图像又被细分到了多个子文件夹中。stagesepx判定每个子文件夹中的帧图像为一个"稳定区间"。它认为"稳定区间"中的图像内容是处于"非变化状态"的。"stable_frame"文件夹中每个"稳定区间"内容如下：![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/mp4_to_frame.jpg)

这时候大家肯定有个疑惑，很多看起来稳定的区间，为什么会被拆分开。例如文件夹0、1和2，看起来都处于"手机桌面"阶段，看起来多帧图像间是没有变化的，但是却被切分成3个稳定区间。那是因为，在操作App过程中，很多细微的变化，是不容易察觉的。但是当我们将其切分成一帧帧的图像进行比对时，机器很容易就发现帧和帧图像之间的区别了。这里不用着急，具体稳定区间的个数，也就是说"stable_frame"中文件夹的数量、每个子文件夹中帧的数量，都可以通过参数控制。如何使用这些参数我们**稍后介绍**。

### 模型训练

现在我们要使用"stable_frame"中稳定区间内的帧图像，来训练一个模型文件，该模型文件将用于后续的预测工作。在我们调用"cutter.cut(video)"方法后，视频文件被切分到了上面的8个文件夹中。正如上面所说，这个切分结果并不是很让人满意，它将视频切得"太细"了。这时候我们可以对切分结果进行"人工调整"。如果某个本应该被划分在同一个稳定区间的帧图像被分在了多个文件夹，我们可以人工将其合并到一个文件夹下。人工调整过程如下：![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/manual_regulation.jpg)

随后对更新后"stable_frame"文件夹内的帧图像进行训练，得到模型文件。

	from stagesepx.classifier.keras import KerasClassifier
	
	# 训练模型文件
	cl = KerasClassifier(
	# 训练轮数
	epochs=10
	)
	cl.train('./stable_frame')
	cl.save_model('./model.h5', overwrite=True)


### 预测

得到模型文件后，我们将其存储到本地。后续我们将使用该模型，预测其他视频文件中，App的启动耗时。预测过程如下： 

	# 使用Keras方法进行预测
	cl = KerasClassifier()
	cl.load_model('./model.h5')
	
	# 将视频切分成帧
	file_name = './mp4_for_forecast.mp4'
	video = VideoObject(file_name)
	# 新建帧，计算视频总共有多少帧，每帧多少ms
	video.load_frames()
	# 压缩视频
	cutter = VideoCutter()
	# 计算每一帧视频的每一个block的ssim和psnr。
	res = cutter.cut(video)
	# 判断A帧到B帧之间是稳定还是不稳定
	stable, unstable = res.get_range()
	# 把分好类的稳定阶段的图片存本地
	res.pick_and_save(stable, 20, to_dir='./forecast_frame', 
		meaningful_name=True)
	# 对切分后的稳定区间，进行归类
	classify_result = cl.classify(file_name, stable, 
		keep_data=True)
	result_dict = classify_result.to_dict()
	
	# 打印结果
	print(result_dict)
	
	# 输出html报告
	r = Reporter()
	r.draw(classify_result, './result.html')


"result_dict"字典中存储了预测结果。查看这个字典，我们可以看到字典总共有4个key，分别是-3、0、2和3。视频文件被切分出来的所有帧视频，均被归属到这四个key下。Keras算法认为预测时归属在key = 0中的帧，和训练模型时文件夹0中的帧属于同一类。同理归属到key = 1中的帧，和训练模型时文件夹1中的帧属于同一类。至于处于不稳定状态的帧，则归属到key = -3中。

有一个很奇怪的地方，经过Keras算法预测后，没有帧被分到了key = 1中。key = 1中的帧，本应是用来表示用户点击App这个行为的帧，但是这些帧却被划分到了key = 2中。这是由于用户点击App这个操作太细微，导致该操作并未被划分到正确的key中。后面我们将介绍通过**调整入参**，修正分类结果。

通过avidemux软件，我们可以看到视频中每一帧图像的相对时间。我们可以看到B点的时间点为"1.916s"，D点的时间点为"5.133s"。从"result_dict"中，我们也可以轻易的B和D点的时间点。但是显然这个让人不满意的分类结果，无法让我们有规律的在"result_dict"中找到B和D点的位置。究竟该如何才能自动化的找到B和D点的位置？

![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/forecast_without_param.jpg)

### 影响稳定区间定的参数

在将视频文件切分成帧图像时，为了能得到跟准确的切分结果，我们需要调整视频切割时的入参，主要参数如下：

| 参数名    | 作用                                                         | 默认值 | 取值范围 |
| :-------- | :----------------------------------------------------------- | :----- | :------- |
| threshold | 判定某一帧图片是否稳定的阈值。阈值越高，则某一帧越难被认定是稳定的。 | 0.95   | [0, 1]   |
| offset    | 补偿值。和threshold相互影响。<br/>有时候设置太高的threshold，会将一个变化很小的区间，切分成多个稳定区间。<br/>通过设置offset值，可以将变化不大的多个稳定区间，连在一起。 | None   | [0, +∞)  |
| block     | 判定某一帧视频的稳定程度时，将帧图像切分的程度。该值越大，计算出的ssim值越敏感。<br/>block = 2，则切分成4宫格。1则不切割。 | 2      | [1, +∞)  |

在不同的场景下，得到最优稳定区间判定结果的入参均有细微区别。为了得到一个比较满意的"国元点金”App启动场景的稳定区间划分结果，通过大量实验，我发现使用下面的参数，可以得到较好的判定结果，并能很好的控制预期和实际结果之间的偏差率：

	file_name = './mp4_for_forecast.mp4'
	video = VideoObject(file_name)
	video.load_frames()
	cutter = VideoCutter()
	# 根据经验，一般想要将"用户点击"这个行为，判定成一个独立的稳定区间，需要设置block = 6和threshold=0.97
	res = cutter.cut(video, block=6)
	# 这里的阈值threshold和切分程度block设置的很高。很容易将变化不大的一段视频切分成很多个稳定区间。设置offset=3，可以将变化不大的多个稳定区间合并成一个。
	stable, unstable = res.get_range(threshold=0.97, offset=3)

通过上述参数控制，预测时切分出来的稳定区间如下：![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/get_section_with_param.jpg)

我们再看"result_dict"中的结果，"点击App"这个用户行为被划分到了"key = 1"中，且为该key下的第一个value值。同时我们可以通过找"key = 2"的第一帧图像的时间点，作为打开App首页的时间点。通过avidemux，我们可以看到该Mp4视频B点的实际值为1.299s，仅和预测值相差0.0001ms，D点实际值为4.333s，误差为0ms。可见预测结果还是让人非常满意的。![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/forecast_with_param.jpg)

通过录制的多个"打开国元点金App"的视频，在使用[ threshold= 0.97, offset = 3, block = 6 ]切分视频、判定稳定区间时，均能将"key =1"的第一帧图像认定为B点，将"key=2"的第一帧认定为D点。这样我们通过取"result_dict"中上述两个位置的value，就可以自动化的得到B和D点的值，从而**自动**计算出"App启动耗时"。

### 选择入参，优化预测结果和精度

上面已经给出了threshold、offset、block参数的定义。但是怎么组合，才能获取到最小的偏差率。我采取的方法是依次尝试几种最常用的参数组合，根据偏差量选定最优入参。

	# 常用参数如下
	# param_list = [门限值：threshold, 偏移量：offset, 切分块数：block ]
	param_list = [[0.97, 1, 6], [0.96, 1, 6], [0.96, 1, 8], [0.97, 1, 8]
	              [0.97, 2, 6], [0.96, 2, 6], [0.96, 2, 8], [0.97, 2, 8]
	              [0.97, 3, 6], [0.96, 3, 6], [0.96, 3, 8], [0.97, 3, 8]]

我们对上述场景的23个视频进行试验，我们事先统计了实际的B和D点的时间。通过不同的参数组合，我们去预测其B和D点的时间，并计算预测和实际启动耗时的偏差率和偏差率。在传入 [ threshold= 0.97, offset = 3, block = 6 ]等参数组合时，预期和实际的App启动耗时偏差量均值仅为4.261ms，偏差量仅为0.142%。这样的数据误差还是很棒的。我们可以使用该组参数，预测预测上述场景的启动耗时。不同入参的偏差量和偏差率数据如下：![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/CSV_Launch_APP.jpg)

 在入参为 [ threshold= 0.97, offset = 3, block = 6 ] 场景下，被预测的23个视频的偏差量和偏差率数据如下：**(改下数字)**![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/CSV_Lunch_APP_0.97_3_6.jpg)

# 专题

## 如何录制视频

我常用两种方式录制视频如下：

- 手机录屏。本文所有的测试视频均是使用手机录制的，录屏相关设置如下：

![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/RecordScreen_param.jpg)  

- abd录屏。使用adb录屏，可以为后期通过Appium等方法，实现自动化批量录制视频提供保障。
	
	```
	:: 使用adb方法录制一个10s的视频
	adb shell screenrecord  --time-limit 10 /sdcard/demo.mp4
	```

但是华为手机无法使用该方法进行录屏，华为ROM阉割了screenrecord功能。

## 如何初始化视频

通过手机或者电脑adb录制出的视频，由于软件录制的帧不稳定，存在每一帧的时长不恒定的现象。为了避免该现象影响分析结果，我们在做切割视频之前，需要使用ffmpg对视频进行初始化。方法如下：

	video = VideoObject(video_NAME, fps=60)

或者直接在cmd中调用ffmpeg，(使用该方法，需要将ffmpeg路径添加到环境变量)：

	ffmpeg -i 待初始化视频路径 -r fps 输出文件名  

初始化前、后视频fps变化如下：

![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/ffmpeg_audio.jpg)  

通过初始化视频，能解决后面一帧视频的时间比前面一帧视频的时间还早的问题。但是，在我通过VideoObjec方法对视频进行初始化时，偶尔还是会出现上述问题。固推荐在cmd中将视频进行初始化。

## Hook方法

常用的Hook有IgnoreHook和CropHook。Hook方法。有什么用？

IgnoreHook的作用是忽略Frame中某一区域的变化，不将其作为判定帧和帧之间是否处于稳定状态的依据。上面的例子中，为了能识别到"用户点击App"这个行为，我设置的参数会导致帧图像间只要有细小的变化，就会被被判定为处于不稳定状态。例如：如果导航栏中显示的时间出现变化，则这些帧图像将可能被划分到多个区间中。固我使用了IgnoreHook屏蔽了导航栏区域。

	# 切割视频
	cutter = VideoCutter()
	# 指定的区域会被屏蔽掉
	hook = IgnoreHook(
	    # 默认情况下，所有的坐标都是从左上角开始的
	    # 如果我们需要偏移到右下角，意味着我们需要向下偏移 0.5 * height，向右偏移 0.5 * width
	    # Hook采用两个参数size与offset，分别对应裁剪区域大小与偏移量
	    size=(0.5, 0.5),
	    offset=(0.5, 0.5),
	    # 例如你希望屏蔽掉高度100宽度200的区域，则：
	    # size=(100, 200),
	    # offset=(100, 100),
	    overwrite=True,
	)
	cutter.add_hook(hook)
	res = cutter.cut(video_path)

常用的还有CropHook函数，作用是只将Frame中某一区域的变化，作为判定多个帧图像是否稳定的依据。参数使用方法同上。

在调用上述函数后，还可以通过FrameSaveHook，查看究竟屏蔽或者选择的是帧中哪块区域。结果会保存在用户指定的文件夹下。

```
hook_save_frame = FrameSaveHook('../6_frame_save_dir')
cutter.add_hook(hook_save_frame)
```

显示屏蔽或展示的区域的效果如下：  

![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/IgnoreHook.jpg)  

## 如何更容易的定位用户点击行为
我们在Android系统上录制视频时，一般是用用户点击屏幕出现的小圆点(需在开发者模式中开启"显示点按操作反馈"，或者在录屏时开启"显示屏幕触摸")来定位用户点击事件。而这个小圆点是白色的。为了能更好的定位到这个小白点，建议将被预测的App设置成深色背景色，加大两者颜色的对比。

## 关于compress_rate参数

在将视频文件切割成帧，计算每个帧的ssim的时候可以传入compress_rate参数。该参数可以控制切割时，视频的压缩率。参数取值范围为 (0, 1], 默认值是0.2。

```
cutter = VideoCutter(compress_rate=0.2)
```

这里有个要注意的点，切割视频时不是压缩率越低越好。我们对相同的视频文件，设置不同的压缩率，查看这些帧的ssim值。

| 帧数 |       行为       | compress_rate=0.2的ssim | compress_rate=1的ssim |
| :--: | :--------------: | :---------------------: | :-------------------: |
|  34  |   用户点击屏幕   |         0.9267          |        0.9780         |
|  47  |     中间环节     |         0.9681          |        0.9822         |
|  52  |  竖屏切换到横屏  |         0.7439          |        0.7965         |
|  65  |     中间环节     |         0.8327          |        0.8791         |
| 107  | 横屏展示五日数据 |         0.6889          |        0.8685         |

由上表数据我们可以看出，在compress_rate=0.2时，不稳定状态帧的ssim值更低，这样更有利于发现视频中的不稳定阶段。如果我们设置过高的compress_rate，可能就会将"用户点击屏幕"这个事件的帧和前后帧，一同判定成一个稳定区间。不同的场景可以尝试不同的compress_rate值，从而微调判定某些帧是否处于稳定区间的结果。

## 横屏录制下IgnoreHook方法的参数

在横屏录制场景下，当App的内容处于竖屏状态下时，该帧图像表现为，左边一部分无内容的黑屏，中间是手机竖屏页面，右边还是一部分无内容的黑屏。

此时如果要屏蔽或者选择某一块区域，要把黑色区域的尺寸也算在内。如果只通过中间部分有手机内容图像的尺寸，来选择或者屏蔽部分区域，则会出现被选择的区域和预期不符的情况。

```
hook_test = IgnoreHook(
size=(0.5, 0.1),
offset=(0, 0.4),
overwrite=True
)

cutter.add_hook(hook_test)
```

屏蔽结果如下：

![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/IgnoreHook_Across.jpg)

# 关于其他场景

我们来看下在其他场景下，stagesepx是如何发挥作用的。

## 场景一、直接刷新出新页面的页面切换过程

我们通过国元点金App的"个股详情"页面切换到"五日"页面的过程，来统计"直接刷新出新页面"场景的页面切换耗时。上述场景可以分成以下几个阶段：

- A.展示”个股详情”页面
- B.点击”五日”按钮
- C.展示中间页
- D.展示”五日”数据

我们定义页面切换耗时 = D - B。页面切换主要阶段如下：![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/FST_to_WR.jpg)

我们选择不同的入参，对上述场景的视频进行预测。在传入 [ threshold= 0.97, offset = 1, block = 6 ]等参数组合时，上述场景的预期和实际耗时的偏差均值仅为0.071ms，偏差率仅为0.009%。预测结果是非常精准的。不同入参的偏差量和偏差率数据如下：![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/CSV_FST_2_WR.jpg)

在入参为 [threshold= 0.97, offset = 1, block = 6] 场景下，被预测的14个视频的偏差量和偏差率数据如下：![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/CSV_FST_2_WR_0.97_1_6.jpg)

## 场景二、滑动出新页面的页面切换过程

有时候在页面切换过程中，会存在一个页面滑动过程，此时要更加关注参数的选择。如果选择不好参数，可能会将页面滑动过程中的某一时间点，误当做刷新出最终页面的时间点。

我们通过国元点金App"自选股列表"页面切换到"个股详情"页面的过程，分析下该场景的页面切换耗时。上述场景可以分成以下几个阶段：

- A.展示”自选股列表”页面
- B.点击”自选股”
- C.页面滑动过程
- D.展示”个股详情”页面

我们定义页面切换耗时 = C - B。页面切换过程中，主要阶段如下：![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/ZXGLB_to_GGXQ.jpg)

我们选择不同的入参，对上述场景的视频进行预测。从下表的数据可以看出，在页面切换过程中存在页面滑动的场景，比直接刷新出新页面的场景，对参数的选择更加敏感。不同的入参，偏差率分布区间更广。

在传入 [ threshold= 0.97, offset = 1, block = 6 ]等参数组合时，上述场景预期和实际耗时的偏差均值仅为0.111ms，偏差率仅为0.116%。预测结果是非常精准的。不同入参的偏差量和偏差率数据如下：

![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/CSV_ZXGLB_2_GGXQ.jpg)

 在入参为 [threshold= 0.97, offset = 1, block = 6] 场景下，被预测的12个视频的偏差量和偏差率数据如下：![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/CSV_ZXGLB_2_GGXQ_0.96_1_6.jpg)

## 场景三、横竖屏变化的页面切换过程

我们通过国元点金App的"五日分时图"横竖屏切换过程，看下该场景的页面切换耗时分析。上述场景可以分成以下几个阶段：

- A.在竖屏展示”五日分时图”
- B.双击屏幕
- C.在横屏展示”五日分时图”

我们定义页面切换耗时 = C - B。页面切换主要阶段如下：![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/WR_Rotate_Vertical.jpg)

在寻找最优入参过程中，发现使用我们常用的几种参数时，均无法得到令人满意的预测结果。传入常用参数，均存在无法将B点的帧图像分类到正确的key下面的现象，即key = 1下无帧图像。对14个视频文件进行预测，仅有2个视频B点的帧图像被划分到了key = 1下。![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/CSV_WR_Rotate_Vertical.jpg)

我猜测出现该现象主要是因为当App的页面从竖屏旋转到横屏状态后，App页面在录制的视频中，所占的比例太少，导致将稳定区间划分到不同的key时，存在较大误差。

怎么解决该问题？采用横屏录制模式。

在该模式下，App页面旋转前，五日分时图页面内容仅占录制视频的一小部分。旋转后，页面内容在录制的视频中铺将满整个屏幕。横屏录制模式下，上述场景的关键步骤如下：   

![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/WR_Rotate_Across.jpg)   

使用横屏录制，使用我们常用的几种参数组合，在14个被测视频中，有13个视频的"用户双击屏幕"这个行为的帧图像被分类到正确的key中。且横竖屏切换的预期和实际耗时偏差率仅为0.673%。预测结果是非常精准的。![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/CSV_WR_Rotate_Across.jpg)

问题虽然解决了，但是我一直不确定原理是啥。欢迎大家集思广益，指点迷津。

## 场景四、存在未知中间步骤的页面切换过程

前面介绍的场景，均是通过固定流程，从页面A跳转到页面B。但是如果跳转过程不固定，存在中间过程C，且C还变化的。stagesepx能否统计这种场景下的页面切换耗时呢？

我们以B站App为例，打开B站时，启动页有时不会弹出广告，有时会弹出静态页面广告，有时甚至还会弹出动态视频广告。面对这个问题，我们该如何统计页面切换耗时？

我们来看下启动B站App可以被分为哪几个阶段：

- A.显示手机桌面
- B.点击B站APP图标
- C.显示首屏
- D.展示广告(如果有)
- E.跳转到主页

我们定义页面切换耗时 = E - B。页面切换过程中，主要阶段如下：![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/launch_BiliBili_app.jpg)

面对这种有时弹出广告，有时不弹出广告的场景。我们该如何训练模型呢？答案很简单，我们在训练模型时，只需要把各种广告的帧放在一个文件夹下即可。这样stagesepx在使用训练好的模型进行预测时，就能判断出视频中有没有广告页、哪些页面是广告页。

我录制了33个打开B站的视频，其中10个没有广告，10个有静态广告、13个有动态广告。通过上述方法，我们来预测一下App的启动耗时 [threshold= 0.96, offset = 4, block = 4] 。经过试验我们可以看出，预期和实际耗时的偏差均值仅为17.68ms，偏差率均值在0.477%。偏差控制的还是很让人满意的。预测数据如下：

![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/CSV_Launch_BiliBili.jpg)

## 场景五、其他场景

其实使用stagesepx进行耗时统计时，不仅仅只能应用于App平台。使用其进行耗时分析时，我们需要的仅是录制好的高质量的视频文件、定义好用户行为的起始结束点、合理的训练集，就可以统计某个用户行为的耗时。例如，统计画一些图片的耗时等。![](https://github.com/150109514/stagesepx_with_keras/blob/master/image_for_readme/draw_picture.jpg)

# 写在最后

先写这么多，后面想到什么，再补充。

所有源码奉上：

大家在实践过程中遇到什么问题，欢迎交流讨论。
