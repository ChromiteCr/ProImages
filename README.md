# ProImages
![version](https://img.shields.io/badge/version-P2g-blue)
![last commit](https://img.shields.io/github/last-commit/ChromiteCr/ProImages)
![commit activity](https://img.shields.io/github/commit-activity/m/ChromiteCr/ProImages)
![stars](https://img.shields.io/github/stars/ChromiteCr/ProImages)
![license](https://img.shields.io/github/license/ChromiteCr/ProImages)

把手机或运动相机拍的照片，通过AI模型和物理光学/胶片建模处理成更接近专业相机直出的效果。处理链路包含四个模块：

1. **降噪**（`proimages/core/denoise`）：目标是用预训练的SCUNet（盲去噪，对手机JPEG泛化好）分块处理手机小底传感器的高噪点；没装torch时用亮度引导的色度降噪兜底，并按估计出的噪声水平决定降多少
2. **HDR/动态范围**（`proimages/core/hdr`）：目标是用边缘保持的局部增益图提亮阴影、保住高光——实测单图合成曝光再做Mertens融合会压暗阴影、把过曝白变灰，所以不用它
3. **景深虚化**（`proimages/core/depth_bokeh`）：目标是用Depth Anything V2 Small做单目深度估计，再用物理光学模型（薄透镜弥散圆随景深和光圈变化、光圈叶片形状、线性光下分层合成保证遮挡边缘正确）渲染虚化，而不是简单高斯模糊
4. **LUT调色与物理相机/胶片效果**（`proimages/core/physical_fx`）：应用行业标准`.cube` LUT文件调色，再叠加物理胶片颗粒、halation（高光光晕）、高光滚降、镜头暗角（cos⁴渐晕）等物理效果。这个模块内部是一条可扩展的"物理效果链"（`effects.py`里的`STAGES`列表），每个效果都是独立文件、独立函数，以后要加色差（chromatic aberration）、镜头光晕（lens flare）、衍射星芒等新的物理效果，只要新增一个文件并加进`STAGES`列表即可，不需要改这个模块之外的任何代码

除了这四个处理模块，还有一个**AI生成LUT**模块（`proimages/core/lut_gen`）：用户填入自己的模型名称和API Key，用一句话描述想要的风格、或丢一张参考图，就能生成一个`.cube`文件供上面第4个模块使用。详见下方"AI生成LUT"一节。

`denoise`、`hdr`、`depth_bokeh`三个模块还没有实现，管线目前不调用它们。它们以后都默认关闭：每个模块落地时在管线参数对象`ProcessOptions`里加上自己的子选项，调用方设置了这个子选项，管线才会运行这一步。`physical_fx`里的5个效果——LUT、颗粒、halation、滚降、暗角——现在全部是真实算法（见下）。

### LUT应用（`physical_fx/lut.py`）

`apply_lut(image, lut_path)`用[colour-science](https://www.colour-science.org/)的`colour.read_LUT()`读取`.cube`文件（自动识别LUT尺寸和定义域），`LUT3D.apply()`做三线性插值把LUT套到图片上。LUT只有在调用方传入`lut_path`时才会执行，不传就跳过这一步（CLI用`--lut`，API用`/v1/jobs`的`lut`上传字段）。

### 颗粒（`physical_fx/grain.py`）

`apply_grain(image, intensity=0.04, grain_size=1.0, rng=None)`不是简单地叠加随机噪声，而是模拟两个胶片颗粒的物理特征：

- **颗粒可见度随曝光变化**：真实胶片的颗粒在中间调最明显，在纯黑/纯白处几乎看不见（对应胶片特性曲线在两端的响应饱和）。代码里用`4 * L * (1 - L)`这个抛物线权重（`L`是像素亮度，取值[0,1]），在`L=0.5`时权重最大为1，在`L=0`或`L=1`时权重为0
- **颗粒尺寸/成团**：先生成逐像素独立的高斯白噪声，再用`scipy.ndimage.gaussian_filter`做空间模糊，让相邻像素的噪声相关起来，形成类似真实卤化银颗粒"成团"的视觉效果，`grain_size`控制这个模糊半径（对应颗粒物理尺寸）

`intensity`控制颗粒强度，`rng`传入`numpy.random.Generator`可以让结果可复现（不传则每次调用结果都不同）。luminance计算复用了`core/color`（见下）。

### Halation（`physical_fx/halation.py`）

`apply_halation(image, threshold=0.75, radius=8.0, intensity=0.35)`模拟高光在片基里散射后重新曝光周围乳剂形成的红橙色光晕：先按亮度阈值`threshold`软性提取高光区域（`highlight_mask`，低于阈值的地方是0），再用`scipy.ndimage.gaussian_filter`（`sigma=radius`）把这个高光蒙版做大半径模糊模拟光在片基里的散射，最后乘上暖色调`HALATION_TINT=(1.0, 0.45, 0.25)`（红>绿>蓝，对应胶片anti-halation染料层的光谱特性）叠加回原图。

### 高光滚降（`physical_fx/rolloff.py`）

`apply_rolloff(image, strength=0.5)`模拟胶片特征曲线（H&D曲线）两端平、中间陡的形状：用平滑阶跃函数`smoothstep(x) = x²(3-2x)`——它在`x=0`和`x=1`处导数为0（对应曲线的趾部toe和肩部shoulder，压缩阴影/高光细节不至硬切），在`x=0.5`处导数最大（对应中间调反差最强），`strength`控制在原图和完整smoothstep之间的混合比例（0=不变，1=完整曲线）。

### 镜头暗角（`physical_fx/vignette.py`）

`apply_vignette(image, strength=0.6)`按cos⁴渐晕定律模拟自然渐晕：以图像中心为原点算每个像素到中心的归一化半径`radius`（中心=0，最远角=1），乘以`strength`近似成入射角`θ`，衰减系数为`cos(θ)⁴`，画面中心不受影响（`θ=0`时衰减=1），向四角逐渐变暗。

## AI生成LUT（`core/lut_gen`）

上面的`apply_lut()`只能**消费**已有的`.cube`文件。这个模块负责**生产**：用户提供自己的模型名称和API Key，用一句话描述风格（"温暖的夏日胶片感"）或丢一张参考图，得到一个`.cube`。

### 为什么是"AI输出参数、Python烘焙LUT"

LLM不能直接输出LUT。一个33³的`.cube`有35,937个RGB三元组、约10万个浮点数，逐token生成既慢又贵，而且几乎必然不平滑——相邻格点跳变会让成片出现色块断层。

所以这里的分工是：**LLM只输出一小组调色参数（JSON），Python用这些参数确定性地烘焙出`.cube`**。这也正好和iPhone的"摄影风格"二维调色盘同构——那个盘本质上就是3个数字驱动一个参数化调色模型。于是AI和调色盘共用同一套参数：AI给初始值，用户拖盘微调，两者走同一个烘焙函数。

### 调色模型：ASC CDL v1.2为底层，lift/gamma/gain作控件

lift/gamma/gain（LGG）是调色行业通用的控件约定，但它本身**没有正式标准**——Resolve、Nuke、Unity等各家的具体公式和中性值写法并不一致。真正发布的标准是**ASC CDL v1.2**（美国电影摄影师协会的Color Decision List：slope/offset/power + saturation），Resolve、Baselight、Nuke（经OpenColorIO）都能读写。所以本模块这样分工：

- **控件用LGG的Nuke Grade形式和标准中性值**：逐通道`out = ((gain − lift)·x + lift)^(1/gamma)`，中性值lift 0 / gamma 1 / gain 1——这是调色师和训练过相关语料的模型都熟悉的写法
- **底层数学就是ASC CDL**：上式与CDL的`(x·slope + offset)^power`精确等价（slope = gain − lift，offset = lift，power = 1/gamma）。白平衡和对比度被安排在幂运算之前、明度折算进幂次，所以**整套调色恰好等于一个CDL**，可以导出成`.cc`文件在Resolve/Nuke里复现同一个调色

### `GradeParams` v2字段

`lut_gen/params.py`，`schema_version`为2，多余字段一律拒绝（模型写了CDL的`slope`之类不会被静默忽略）：

| 字段 | 中性值 | 范围 | 含义 | 对应调色盘 |
|---|---|---|---|---|
| `lift` [r,g,b] | 0 | 各−0.5..0.5 | 输出黑位 | — |
| `gamma` [r,g,b] | 1 | 各0.5..2 | 中间调，>1变亮（CDL的power是1/gamma，方向相反） | — |
| `gain` [r,g,b] | 1 | 各0..2，且逐通道gain ≥ lift | 输出白位 | — |
| `saturation` | 1 | 0..2 | CDL饱和度（按Rec.709亮度），0为全灰 | **X轴** = saturation − 1 |
| `contrast` | 1 | 0.25..2 | 绕固定的0.5转轴 | — |
| `tone` | 0 | −1..1 | 整体明度，折算为整体gamma × 2^tone，黑白两端不动、不裁切 | **Y轴** |
| `temperature` | 0 | −1..1 | 冷↔暖 | **下方滑块** |
| `tint` | 0 | −1..1 | 绿↔品红 | — |
| `name` / `description` | — | 80 / 400字 | 风格名和说明，合并空白、去掉引号，写进`.cube`注释和`.cc` | — |

三个轮里RGB取相同值就是调整该区间的亮度，取不同值就是给该区间加色偏。`temperature`/`tint`没有行业标准，是本项目的约定：通道增益按Rec.709亮度归一，保证中灰亮度不变（饱和色的亮度仍会略变）。

运算顺序和折算成CDL的公式（`lut_gen/cdl.py`，`wb`为白平衡增益，`k`为contrast）：

```
slope  = k · (gain − lift) · wb
offset = k · lift + 0.5 · (1 − k)
power  = 1 / min(gamma · 2^tone, 2)
sop    = clamp01(x · slope + offset) ^ power
out    = clamp01(luma + saturation · (sop − luma))      # luma = Rec.709(sop)，即ASC CDL v1.2正向公式
```

`.cc`是这套调色的精确表示（用OpenColorIO读回，与本项目计算结果相差约1e-5）；`.cube`是在33³网格上的采样近似。三线性插值跟不上两类拐点：幂次小于1（提亮中间调）时最暗一格里很陡的曲线，以及钳位造成的折角。在样张上的典型调色，`.cube`与精确结果最大差约3/255、平均约0.03/255；极端参数组合在最暗处可达约27/255。为此有效gamma（gamma × 2^tone）封顶为2，`.cc`和`.cube`用的都是封顶后的值；需要精确结果时用`.cc`。

**v1 → v2是不兼容升级**：v1的lift/gamma/gain是中性值为0的自定义偏移量（再乘上0.25/0.5/0.35这样的隐藏系数），saturation、contrast也以0为中性。模型按行业习惯写出`gamma=[1,1,1]`（意为不变）会被当成强烈调色，写`gain=[1.05,1,0.95]`直接校验失败。v2全部改成标准写法；带`schema_version: 1`的参数文件会被明确拒绝，`/v1/luts/bake`和`proimages-lut --params`还要求参数里必须写出`schema_version`，防止缺省字段的旧参数被当成v2读错。

### 模块内部

- **`cdl.py`**：`to_cdl(params)`把整套参数折叠成一个CDL；`apply_cdl(rgb, cdl)`是ASC CDL v1.2正向公式（先钳位再取幂、按SOP结果的Rec.709亮度做饱和度，亮度复用`core/color.luminance()`，其权重正是CDL规定的Rec.709）；`to_cc_xml()`用标准库写出`.cc`
- **`grade.py`**：`apply_grade(rgb, params) = apply_cdl(rgb, to_cdl(params))`，对任意形状为`(..., 3)`的数组生效——LUT的identity表是`(N,N,N,3)`、照片是`(H,W,3)`，"烘焙LUT"和"直接调色一张图"共用同一份实现
- **`bake.py`**：用`colour.LUT3D.linear_table(size)`拿到identity表，过一遍调色，写成`.cube`；注释里记下`schema_version 2`和对应的CDL数值，LUT自带说明。`colour.write_LUT`只接受文件路径，需要文本时走临时文件再读回
- **`image_stats.py`**：`extract_color_stats(image)`按亮度分位数把像素切成阴影/中间调/高光三段分别算平均RGB，再算冷暖倾向、品红↔绿倾向（与`tint`同号）、饱和度、对比度、逐通道p1/p50/p99，以及一个粗略的CDL拟合（假设参考图场景中性且满幅，从p1/p50/p99反推lift/gamma/gain）。**参考图这条路径不让模型"目测"色温**——先用算法量出客观数字，模型只负责把数字翻译成风格化参数
- **`llm.py`**：OpenAI兼容客户端，走JSON输出模式。只实现这一套协议是因为它能覆盖OpenAI、DeepSeek、Moonshot、通义千问、智谱以及本地ollama/vLLM，用户只需改`base_url`。模型输出校验失败时会带上错误信息重试一次，仍失败才报错
- **`prompts.py`**：system prompt写明公式、每个参数的中性值与典型幅度、gamma与CDL power方向相反、等值RGB等于亮度调整，并给出完整的JSON示例

### API Key的处理

Key由调用方每次传入（CLI参数或API请求体），**服务器不存盘、不进日志、不写进`JobStore`**，只在单次请求的生命周期内存在。`lut_routes.py`里模型调用失败时只回报异常类型、不回显请求内容——错误消息是最容易泄露Key的地方，有一个专门的测试用例（`test_generate_error_response_does_not_leak_the_api_key`）盯着这条。

## 项目结构

```
proimages/
  core/
    system/io.py     load_image(path)/decode_image(bytes, filename)统一解码成[0,1]浮点RGB numpy数组：
                      JPEG/PNG等走Pillow并按EXIF方向摆正（手机竖拍照片不会横着出来）；
                      HEIC/HEIF走heif组的pi-heif；RAW（.dng/.cr2/.cr3/.nef/.arw/.raf/.rw2）走heavy组的
                      rawpy，用拍摄时的白平衡（文件里没有才退回日光）和sRGB曲线显影；缺少对应组时
                      报错指明该装哪个组。16位灰度图按真实亮度读取（Pillow默认转换会把它截成全白）；
                      读不出来的文件统一报ValueError("not a readable <格式> file")，不再把对象地址
                      之类的内部信息带进API的任务状态。save_image()/encode_png()编码回8位图片
    system/tiling.py tiled_apply(fn, image, tile=512, overlap=64, multiple=64)：整图先reflect补边
                      （边缘像素和内部像素一样有上下文），切成相互重叠的块逐块交给fn（块的宽高都是
                      multiple的倍数，传进去的是副本，fn可以原地改），再用升余弦羽化拼回去——相邻两块
                      的权重在重叠区相加恰好为1，所以没有接缝；numpy进numpy出，fn可以改变通道数。
                      P3的SCUNet分块推理用它，12MP时额外占用约555MB
    denoise/          降噪模块，尚未实现（P3），管线不调用
    hdr/               HDR模块，尚未实现（P4），管线不调用
    depth_bokeh/      景深虚化模块，尚未实现（P5/P5a），管线不调用
    physical_fx/      LUT调色与物理相机/胶片效果模块，process(image, lut_path=None)->image
      __init__.py     process()：lut_path给定时先调apply_lut()，再依次跑STAGES里的4个效果
      effects.py      STAGES列表：[apply_grain, apply_halation, apply_rolloff, apply_vignette]
                      （apply_lut签名不同，在__init__.py里单独处理，不在这个列表里）
      lut.py           apply_lut(image, lut_path)->image，用colour-science读.cube并三线性插值应用
      grain.py         apply_grain(image, intensity=0.04, grain_size=1.0, rng=None)->image，
                      中间调加权+空间相关噪声模拟胶片颗粒
      halation.py      apply_halation(image, threshold=0.75, radius=8.0, intensity=0.35)->image，
                      高光提取+大半径模糊+暖色调叠加模拟halation光晕
      rolloff.py       apply_rolloff(image, strength=0.5)->image，smoothstep S形曲线模拟胶片特征曲线
      vignette.py      apply_vignette(image, strength=0.6)->image，cos⁴渐晕定律模拟镜头暗角
    lut_gen/          AI生成LUT模块（生产.cube，与physical_fx/lut.py的"消费"相对）
      params.py       GradeParams v2（pydantic模型）：LGG标准中性值、各字段范围、
                      gain ≥ lift校验、名称清洗、拒绝多余字段和旧版schema
      cdl.py          CDL模型；to_cdl()折叠参数、apply_cdl()按ASC CDL v1.2正向公式计算、
                      to_cc_xml()导出.cc
      grade.py        apply_grade(rgb, params)->rgb = apply_cdl(rgb, to_cdl(params))，
                      对(...,3)任意形状生效，烘焙LUT和调色图片共用这一份
      bake.py         bake_lut()/bake_cube_file()/bake_cube_text()，
                      identity表过调色再写成.cube，注释里带schema_version和CDL数值
      image_stats.py  extract_color_stats(image)->dict，按亮度分位数拆阴影/中间调/高光
                      分别统计，另给逐通道分位数和粗略CDL拟合，供参考图路径喂给模型
      llm.py          OpenAI兼容客户端，params_from_description()/params_from_reference_image()，
                      校验失败时重试一次
      prompts.py      system prompt（公式、中性值、JSON示例）和两条入口各自的user prompt模板
    color/            色彩空间共用工具
      __init__.py     luminance(image)->ndarray按Rec.709权重算亮度，grain.py、halation.py、
                      lut_gen/grade.py、lut_gen/image_stats.py、noise.py都复用它
      transfer.py     srgb_to_linear()/linear_to_srgb()：IEC 61966-2-1的sRGB分段曲线（解码阈值0.04045、
                      编码阈值0.0031308），全程float32，12MP三通道约0.11s；不依赖colour-science（它输出
                      float64且更慢，只在测试里当对照，两者相差<1e-6）。负值走线性段、大于1走幂段、
                      不做裁切，任何有限输入都不出NaN。P4要在线性光里乘增益，靠它进出线性光
    filters.py        边缘保持滤波，P3色度降噪、P4底层提取、P5深度精修共用：
                      box(x, r)：(2r+1)²窗口均值，边界按半像素对称反射；
                      guided_filter(guide, src, r, eps)：He、Sun、Tang的引导滤波，灰度引导用标量公式，
                      彩色引导用3×3协方差形式（逐像素闭式求逆，向量化，不走Python循环；求逆这一步
                      在系数分辨率上用float64做——接近中性灰的彩色引导图在小eps下协方差矩阵几乎
                      奇异，float32求逆在eps=1e-6时误差从0.5左右到几十不等，float64下在1e-4量级）；
                      guided_filter按全分辨率计算，12MP彩色引导要好几GB内存，整张照片请用下面两个；
                      fast_guided_filter(guide, src, r, eps, s)：在1/s分辨率上拟合系数、双线性放大后
                      作用到全分辨率引导图上（12MP灰度、s=4、r=16约54ms）；
                      guided_upsample(guide, src, r, eps)：把低分辨率的图（比如模型出的深度图）借
                      全分辨率彩色照片的边缘放大回原尺寸（378×504→12MP约0.15s），同亮度的红绿交界
                      这种只有颜色差别的边缘也能保住。计算前先减去全局均值，避免float32下
                      var=E[I²]−E[I]²的相消误差（log2亮度约−8时尤其明显）；输出一律float32
    noise.py          estimate_noise(image)->float：Donoho的MAD噪声估计——取最细一层Haar小波的对角子带
                      HH，σ=median(|HH|)/0.6745，在显示编码的Rec.709亮度上算，是相对尺度、不含色度
                      噪声（三通道各自独立的噪声σ在亮度上表现为约0.749σ），12MP约40ms。中位数对边缘
                      不敏感，只有纹理很密的画面会偏高；反过来，大片平坦或被裁切（过曝、死黑）的区域
                      HH恰好为0，会把中位数往下拉——σ=0.012的噪声在平坦区占20%/30%/40%时读成
                      0.0089/0.0066/0.0038，超过一半时直接是0。P3按它决定降噪强度、P4按它收小增益，
                      用之前都要先把这些区域排除掉
    options.py        ProcessOptions（pydantic模型）：选择要跑哪些可选阶段。每个阶段落地时才加入
                      自己的子选项字段，设置了该字段就开启这一步，所以不会出现"开了却没实现"的
                      选项；拒绝未知字段；不含任何文件路径（API从任意客户端接收这个JSON，带路径就
                      等于让请求指挥服务器读写文件，文件一律走上传）。目前还没有字段，只接受{}
    pipeline.py       process_image(image, lut_path=None, options=None, device=None)：可选阶段按
                      denoise→hdr→depth_bokeh的顺序、只在options设置了对应子选项时运行，最后一律
                      交给physical_fx.process()（lut_path透传）。三个可选阶段都还没实现，所以现在
                      等同于直接调physical_fx——有测试固定随机种子逐位核对这一点。device是需要
                      模型的阶段在哪个设备上跑（cuda/mps/cpu），None表示由阶段自己探测
  gpu_config.py       detect_device()探测cuda/mps/cpu，支持传override参数强制指定；
                      torch是惰性import，未安装（基础安装）时返回"cpu"，
                      这样基础安装也能起完整API而不是在启动时崩掉
  model_download.py  ensure_model(repo_id)用huggingface_hub.snapshot_download把预训练权重下载到
                      ~/.cache/proimages/models/（可用PROIMAGES_MODELS_DIR环境变量改路径），已存在则直接复用缓存
  cli_args.py         add_device_argument()定义共用的--device参数，CLI和API服务入口都调用它
  cli.py              proimages命令行入口：读入一张图片路径→跑process_image()→写到输出路径；
                      探测到（或--device指定）的设备会传给process_image()
  lut_cli.py          proimages-lut命令行入口：--describe/--reference/--params三选一，
                      生成或烘焙出.cube，--cdl-out另存.cc；--params路径不调LLM也不需要Key
  api/
    app.py            FastAPI应用实例，挂载GET /health、jobs路由和luts路由
    lifespan.py       FastAPI启动时用detect_device()探测设备存到app.state.device，
                      创建一个JobStore实例存到app.state.job_store，按PROIMAGES_MAX_JOBS（默认1）
                      创建任务槽位app.state.job_slots（asyncio.Semaphore），并用app.state.job_tasks
                      持有后台任务的引用（事件循环对任务只保留弱引用）。PROIMAGES_MAX_JOBS不是
                      正整数时服务直接启动失败并说明原因
    server_cli.py     proimages-api命令行入口：解析--host/--port/--device，用uvicorn跑api.app:app
    jobs/
      models.py       JobStatus枚举（pending/running/completed/failed）和JobRecord数据类
                      （job_id、状态、创建时间、结果字节、错误信息）
      store.py        JobStore：内存态的{job_id: JobRecord}字典，所有读写都加asyncio.Lock
      runtime.py      run_job()：先等一个任务槽位（等待期间状态保持pending），拿到后标成running，
                      用decode_image()按上传文件名解码（与CLI同一套EXIF/HEIC/RAW处理）→带着options
                      和服务器设备跑process_image()→编码成PNG字节→写回JobStore；解码、处理、编码
                      都在槽位内完成，所以同时解码展开在内存里的照片不超过PROIMAGES_MAX_JOBS张（排队
                      的任务仍各自持有上传的原始字节，排队长度不设上限；做完的PNG结果也一直留在
                      内存里的JobStore中，目前没有清理）。异常会被捕获并记成failed状态，而不是让
                      请求本身报错
    http/
      routes.py       三个接口：POST /v1/jobs（上传文件，可选的lut上传和options字段——ProcessOptions
                      的JSON字符串；创建任务并用asyncio.create_task后台跑，立即返回job_id。options
                      由FastAPI在进入处理函数之前校验，非法JSON、未知字段一律422，不会先建任务、
                      写临时LUT文件再失败）、GET /v1/jobs/{job_id}（查状态，不存在返回404）、
                      GET /v1/jobs/{job_id}/result（状态不是completed时返回409，是的话返回PNG字节）
      lut_routes.py   两个接口：POST /v1/luts/generate（传describe或reference + model/api_key
                      /base_url，模型调用放在工作线程里不阻塞服务，返回GradeParams、CDL数值、
                      .cc文本和烘焙好的.cube）、POST /v1/luts/bake（只传GradeParams，不调LLM
                      不需要Key，毫秒级——调色盘拖完后调的就是它）；两者的size都限2..64。
                      参考图在调用模型之前解码：读不出来返回422，缺少解码所需的可选组（如用
                      DNG/HEIC做参考图却没装heavy/heif）返回501并写明该装哪个组，都不会被
                      误报成"模型请求失败"
```

LUT这两个接口不走上面的job异步机制，请求直接返回结果：返回体是小JSON、没有GPU排队问题、模型输出几十个数字通常2-5秒就回来，调用方在按钮上转个圈即可。`JobStore`/`JobRecord`保持只服务图像处理任务。

CLI和API服务共用同一份`core/`处理逻辑，不存在"CLI版本"和"API版本"两套算法代码。

## 依赖与运行

用[uv](https://github.com/astral-sh/uv)管理依赖。依赖分成两层，因为LUT这条链路完全用不到深度学习相关的重依赖：

| | 包含 | 装出来的虚拟环境 | 能用什么 |
|---|---|---|---|
| **基础安装** | fastapi、uvicorn、python-multipart、numpy、pillow、colour-science、scipy、openai、pydantic | ~220MB | LUT全链路：`core/lut_gen`生成LUT、`core/physical_fx`全部5个效果、完整API（含`/v1/luts/*`和`/v1/jobs`）、三个CLI入口 |
| **`heavy`可选组** | 额外加torch、rawpy、huggingface_hub | ~800MB+ | 再加上RAW格式输入，以及`core/denoise`、`core/hdr`、`core/depth_bokeh`里需要预训练模型的部分（这三个模块尚未实现，管线目前不调用） |
| **`heif`可选组** | 额外加pi-heif | +约1MB | 读取HEIC/HEIF（iPhone默认的拍照格式） |

`heif`组刻意用只含解码器的pi-heif而不是pillow-heif：两者接口相同，但pillow-heif的安装包里捆绑了GPL许可的x265编码器，pi-heif只带LGPL的libheif/libde265，更适合这个MIT项目。

```bash
# 只用LUT链路（比如被其他项目当依赖装）
uv sync --extra dev

# 完整安装（RAW/HEIC输入、后续的降噪/HDR/景深模块）
uv sync --extra heavy --extra heif --extra dev
```

`dev`组包含pytest、httpx，以及opencolorio——后者只在测试里用作ASC CDL的参考实现，用来核对`apply_cdl`和导出的`.cc`。

被其他项目作为可编辑依赖引入时，不带extra即可，不会拖进torch：

```bash
uv pip install -e ../ProImages
```

基础安装下`gpu_config.detect_device()`返回`"cpu"`——torch本身就是提供GPU访问的东西，没有它就确实只有CPU，而需要GPU的那三个模块在基础安装下本来也不可用。对RAW文件调用`load_image()`会抛出明确指向`proimages[heavy]`的ImportError，而不是一个裸的`No module named 'rawpy'`；HEIC同理，指向`proimages[heif]`。

```bash

uv run proimages input.jpg output.png --device cpu                       # CLI：处理单张照片，不调色
uv run proimages input.jpg output.png --lut "Example-LUTs/Arabica 12.CUBE"  # CLI：额外套用一个LUT

uv run proimages-api --host 127.0.0.1 --port 8001     # API：起服务
# 另开一个终端：
curl -X POST http://127.0.0.1:8001/v1/jobs -F "file=@input.jpg"                                    # 提交，不调色
curl -X POST http://127.0.0.1:8001/v1/jobs -F "file=@input.jpg" -F "lut=@Example-LUTs/Arabica 12.CUBE"  # 提交，套用LUT
curl -X POST http://127.0.0.1:8001/v1/jobs -F "file=@input.jpg" -F 'options={}'                    # 提交，带管线参数（JSON字符串）
curl http://127.0.0.1:8001/v1/jobs/<job_id>                        # 查状态
curl http://127.0.0.1:8001/v1/jobs/<job_id>/result -o output.png   # 完成后取结果

PROIMAGES_MAX_JOBS=2 uv run proimages-api   # 允许同时处理2个任务（默认1个）

uv run pytest                    # 跑测试
```

API和CLI共用同一套解码：上传RAW或HEIC时，接口按上传文件名的后缀选择解码方式（`curl -F "file=@photo.dng"`会自动带上文件名），同样需要装对应的可选组。

`options`字段是`ProcessOptions`的JSON字符串，不传就是默认管线。目前三个可选阶段都没实现，所以它只接受`{}`；写`{"denoise": {}}`这类还不存在的阶段会直接返回422（`extra_forbidden`），而不是被悄悄忽略。降噪、HDR、景深各自落地时会加上自己的子选项，届时在这里写`{"denoise": {...}}`即可开启。

任务并发：服务器同一时间最多处理`PROIMAGES_MAX_JOBS`个任务（默认1），多出来的任务在`GET /v1/jobs/{job_id}`里显示为`pending`，前一个做完自动开始。限制的是内存而不是CPU：每个在跑的任务都持有解码后的整张照片和各阶段的工作副本，后续的景深渲染器在8MP下一次就要约2.6GB。这个值在服务启动时读取，不是正整数（比如`0`、`two`）时服务直接启动失败并给出原因。

生成LUT（`proimages-lut`）：

```bash
# 用一句话描述风格生成
uv run proimages-lut --describe "温暖的夏日胶片感" --model gpt-4o --api-key sk-xxx -o summer.cube

# 用参考图生成（先算色彩统计再连图给模型）
uv run proimages-lut --reference ref.jpg --model gpt-4o --api-key sk-xxx -o ref_look.cube

# 直接从参数文件烘焙，不调LLM、不需要Key；--cdl-out 同时导出可在Resolve/Nuke里读的.cc
uv run proimages-lut --params params.json -o custom.cube --cdl-out custom.cc

# 指向其他OpenAI兼容厂商
uv run proimages-lut --describe "赛博朋克冷调" --model deepseek-chat \
  --api-key sk-xxx --base-url https://api.deepseek.com/v1 -o cyber.cube

# 加 --save-params 会把AI给的参数也存成JSON，方便手动微调后重新烘焙
```

参数文件（v2，必须写出`schema_version`，没写的字段取中性值）：

```json
{
  "schema_version": 2,
  "name": "Teal Orange",
  "lift": [-0.02, 0.0, 0.04],
  "gamma": [1.0, 1.0, 1.0],
  "gain": [1.08, 1.0, 0.9],
  "saturation": 1.15,
  "contrast": 1.1
}
```

API Key也可以通过`PROIMAGES_API_KEY`环境变量传，省得每次敲。生成出来的`.cube`直接就能给上面的`proimages --lut`用。`--size`（默认33）限2..64。

`--device`不传时会自动探测：有CUDA用cuda，苹果芯片用mps，否则用cpu。CLI把探测结果传给`process_image()`，API服务把启动时探测的`app.state.device`传给每个任务——目前还没有需要模型的阶段，P3的SCUNet降噪、P5a的深度估计会在这个设备上跑。

测试护栏（仓库根目录的`conftest.py`，在pytest启动、收集测试之前就装好，所以测试文件的模块级代码和module/session级fixture也受约束）：

- **不联网**：连接非本机地址的socket（`connect`/`connect_ex`）和非本机域名的解析（`getaddrinfo`/`gethostbyname`/`gethostbyname_ex`）一律抛错。光拦这些还不够——本机设了`HTTP(S)_PROXY=http://127.0.0.1:7897`这类本地代理时，HTTP客户端连的是本机代理端口，由代理替它出网。所以护栏同时清掉`HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`（大小写都清），并把`NO_PROXY`设成`*`（这也让urllib、requests不再回退到macOS的系统代理）。验证护栏的网络断言只用永远不会被路由的192.0.2.1和保留的`.invalid`域名，护栏失效时也连不上任何真实服务器（`.invalid`顶多引发一次注定查不到的DNS查询）；验证下载护栏的测试先确认护栏已经装好，再用一个不存在的仓库名调用，即使用`pytest --noconftest`跳过护栏也不会真的下载权重
- **不下载模型**：`model_download.ensure_model`一调用就抛错；`model_download.MODELS_DIR`指向一个空的临时目录，这样即使某个模块在导入时就拿到了真正的`ensure_model`，也只会缓存未命中、撞上网络护栏，而不会悄悄加载本机已下载的权重。用到模型的测试要自己monkeypatch加载函数
- **基础安装能导入全部模块**：`proimages/guards_test.py`在子进程里把torch、torchvision、rawpy、huggingface_hub、transformers、spandrel、pi_heif都设为不可导入，再按文件逐个导入`proimages`下的全部模块（包括没有`__init__.py`的目录），守住"可选依赖一律在函数里惰性import"这条约束；以后的阶段引入新的重依赖时要加进这个列表

测试用的样张（`Example-imgs/`）和LUT文件（`Example-LUTs/`，35个`.CUBE`文件）只放在本地，`.gitignore`里已排除，不会进版本库。

## 开发规划

已完成：工程骨架（P2）、物理胶片效果链（P2a–P2c）、AI生成LUT（P2d）、依赖拆分（P2d1）。接下来按顺序推进：

| 版本 | 阶段 | 内容 | 状态 |
|---|---|---|---|
| P2e | LGG行业标准化 | GradeParams v2：以ASC CDL v1.2为底层数学，LGG用标准中性值（0/1/1），整套调色可导出`.cc`给Resolve/Nuke | 完成 |
| P2f | 输入层修正 | EXIF方向、HEIC（`heif`可选组）、RAW改用拍摄白平衡与sRGB曲线、统一字节解码 | 完成 |
| P2g | 管线接线与共享工具 | 管线参数对象、设备传递、API的`options`字段与并发上限；引导滤波、sRGB转换、噪声估计、分块推理等共享工具 | 完成 |
| P3 | 降噪 | SCUNet分块推理（heavy）+ 无torch的色度降噪兜底 + 噪声门控 | 待开始 |
| P4 | 动态范围 | 边缘保持的局部增益图：只提亮阴影，过曝区不变灰 | 待开始 |
| P5 | 景深虚化 | 线性光分层FFT物理虚化渲染器 + 外部深度图输入（纯numpy，基础安装可用） | 待开始 |
| P5a | 自动深度估计 | Depth Anything V2 Small（Apache-2.0） | 待开始 |

贯穿全程的约束：

- 新模块默认关闭，不带新参数时处理链路与现在完全一致
- 基础安装不含torch：模型相关代码函数内惰性import，放在`heavy`可选组；渲染器、HDR、色度降噪只用numpy/scipy
- 默认只用可商用许可的模型权重（Apache-2.0 / MIT），非商用权重（Depth Anything V2 Base/Large、Depth Pro）不纳入
- 测试不联网、不下载模型

暂不做：色彩管理（iPhone照片是Display P3，目前按sRGB处理）、NAFNet/Restormer降噪、扩展曝光融合（EEF）强效HDR、虚化的遮挡背景补全与纵向色差、读取iPhone人像HEIC自带的深度图。

## 版本记录

| 版本 | 日期 | 变更内容 | 类型 |
|------|------|----------|------|
| P2g | 2026-10-05 | 管线接线与共享工具：新增管线参数对象ProcessOptions（拒绝未知字段、不含文件路径），process_image接收options和device，CLI与API把探测到的设备传进管线；POST /v1/jobs新增options字段（ProcessOptions的JSON字符串，非法时在建任务之前就返回422）；新增PROIMAGES_MAX_JOBS任务槽位（默认1），超出的任务排队显示为pending；降噪/HDR/景深三个占位模块不再被管线调用，等各自实现后由options开启；新增共享numpy工具：引导滤波、快速引导滤波与彩色引导上采样（core/filters），float32的sRGB转换（core/color/transfer），Haar-MAD噪声估计（core/noise），升余弦羽化的分块处理（core/system/tiling）；测试护栏：pytest收集测试之前就禁止下载模型和联网（本机HTTP代理这条路也堵上），子进程导入图测试守住基础安装不依赖torch等可选依赖 | feat |
| P2f | 2026-10-01 | 输入层修正：按EXIF方向摆正照片（此前手机竖拍会横着出来）；新增heif可选组（只含解码器的pi-heif）读取HEIC；RAW改用拍摄白平衡与sRGB曲线显影（此前是rawpy默认的日光白平衡与BT.709曲线，RAW默认输出因此改变）；修复16位灰度图被读成全白；读不出的文件给出干净的错误信息；新增decode_image()统一CLI、任务接口和参考图的解码，API因此也能收RAW/HEIC；/v1/luts/generate的参考图在调用模型前解码，读不出返回422、缺解码组返回501并写明该装哪个组 | fix |
| P2e | 2026-09-30 | LGG行业标准化：GradeParams升级到v2（不兼容v1），以ASC CDL v1.2为底层数学，lift/gamma/gain改用标准中性值0/1/1，多余字段与旧版schema一律拒绝；整套调色可导出.cc（OpenColorIO读回误差约1e-5）；提示词写明公式、中性值与JSON示例，模型输出校验失败时重试一次；参考图统计新增逐通道分位数与粗略CDL拟合；/v1/luts/generate不再阻塞事件循环，size限2..64；README加徽章与「开发规划」一节 | feat |
| P2d1 | 2026-07-22 | 拆分依赖：torch/rawpy/huggingface_hub移入heavy可选组，基础安装只保留LUT链路所需（虚拟环境800M+降至~220MB）；gpu_config的torch改为惰性import并在缺失时回退cpu，使基础安装也能起完整API；RAW读取缺rawpy时报错指向heavy组 | refactor |
| P2d | 2026-07-22 | 新增AI生成LUT模块core/lut_gen：GradeParams跨仓库契约、参数化调色数学、.cube烘焙、参考图色彩统计、OpenAI兼容客户端；新增proimages-lut命令行入口和/v1/luts/{generate,bake}两个接口 | feat |
| P2c | 2026-07-21 | 实现halation（高光提取+大半径模糊+暖色调叠加）、高光滚降（smoothstep S曲线）、镜头暗角（cos⁴渐晕）三个效果，luminance计算提取到core/color供grain/halation复用；至此physical_fx的5个效果全部实现完毕 | feat |
| P2b | 2026-07-21 | 实现LUT应用（colour-science三线性插值.cube）和物理颗粒（中间调加权+空间相关噪声）两个效果，CLI加--lut参数，API的/v1/jobs加lut上传字段 | feat |
| P2a | 2026-07-21 | core/lut_grain重构为core/physical_fx，内部拆成LUT/颗粒/halation/滚降/暗角5个独立效果文件+STAGES有序效果链，便于后续新增物理效果 | refactor |
| P2 | 2026-07-21 | 完成工程骨架：FastAPI异步任务API + CLI，两者共用core/管线（4个模块均为占位直通），gpu_config设备探测，model_download权重下载占位 | milestone |
| P1 | 2026-07-21 | 初始化项目仓库与README | milestone |


## 许可证

本项目基于 MIT License 开源。详见 [LICENSE](LICENSE) 文件。

```
MIT License

Copyright (c) 2026 zgdhzad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```
