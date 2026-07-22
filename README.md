# ProImages

把手机或运动相机拍的照片，通过AI模型和物理光学/胶片建模处理成更接近专业相机直出的效果。处理链路包含四个模块：

1. **降噪**（`proimages/core/denoise`）：目标是用预训练的Restormer或NAFNet处理手机小底传感器的高噪点，RAW输入会在去马赛克之前先在拜耳域做降噪
2. **HDR/动态范围**（`proimages/core/hdr`）：目标是先用经典exposure fusion算法恢复单张照片的高光/阴影细节，后续再考虑单图HDR重建的深度学习方法
3. **景深虚化**（`proimages/core/depth_bokeh`）：目标是用Depth Anything V2/MiDaS做单目深度估计，再用物理光学模型（弥散圆随景深和光圈变化、光圈形状、遮挡边缘正确合成）渲染虚化，而不是简单高斯模糊
4. **LUT调色与物理相机/胶片效果**（`proimages/core/physical_fx`）：应用行业标准`.cube` LUT文件调色，再叠加物理胶片颗粒、halation（高光光晕）、高光滚降、镜头暗角（cos⁴渐晕）等物理效果。这个模块内部是一条可扩展的"物理效果链"（`effects.py`里的`STAGES`列表），每个效果都是独立文件、独立函数，以后要加色差（chromatic aberration）、镜头光晕（lens flare）、衍射星芒等新的物理效果，只要新增一个文件并加进`STAGES`列表即可，不需要改这个模块之外的任何代码

除了这四个处理模块，还有一个**AI生成LUT**模块（`proimages/core/lut_gen`）：用户填入自己的模型名称和API Key，用一句话描述想要的风格、或丢一张参考图，就能生成一个`.cube`文件供上面第4个模块使用。详见下方"AI生成LUT"一节。

`denoise`、`hdr`、`depth_bokeh`三个模块目前还是占位直通（identity）。`physical_fx`里的5个效果——LUT、颗粒、halation、滚降、暗角——现在全部是真实算法（见下）。

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

### `GradeParams`：跨仓库的公开契约

调色盘UI在**另一个独立仓库**里，通过HTTP消费本仓库的API，所以`GradeParams`（`lut_gen/params.py`）是两个仓库之间的契约，带`schema_version`字段供对方检测不匹配。字段采用调色行业标准的lift/gamma/gain模型（DaVinci等专业工具的通用语言，LLM也熟悉）：

| 字段 | 范围 | 含义 | 对应调色盘 |
|---|---|---|---|
| `tone` | -1..1 | 整体明度 | **Y轴** |
| `saturation` | -1..1 | 饱和度，灰↔纯 | **X轴** |
| `temperature` | -1..1 | 色温，冷↔暖 | **下方滑块** |
| `tint` | -1..1 | 色调，绿↔品红 | — |
| `contrast` | -1..1 | 对比度 | — |
| `lift` / `gamma` / `gain` | 各RGB三元组 | 阴影/中间调/高光的色偏 | — |
| `name` / `description` | str | 风格名和说明，写进`.cube`注释 | — |

前三个字段就是盘的三个自由度。所有数值范围由pydantic强制校验——模型给出越界值时会在解析阶段被挡住，不会流到烘焙环节。UI仓库可以直接从`/openapi.json`生成客户端，字段说明里写明了各自对应盘的哪个轴。

### 模块内部

- **`grade.py`**：`apply_grade(rgb, params)`是纯数学函数，对任意形状为`(..., 3)`的数组生效。这点是刻意设计的——LUT的identity表是`(N,N,N,3)`、照片是`(H,W,3)`，同一个函数都能处理，所以"烘焙LUT"和"直接调色一张图"共用同一份实现，不会两边算法漂移。运算顺序：白平衡(temperature/tint) → lift/gamma/gain → 对比度(绕0.5为轴) → 明度 → 饱和度（复用`core/color.luminance()`）
- **`bake.py`**：用`colour.LUT3D.linear_table(size)`拿到identity表，过一遍`apply_grade()`，写成`.cube`。`colour.write_LUT`只接受文件路径、不能返回字符串，所以需要`.cube`文本时走临时文件再读回
- **`image_stats.py`**：`extract_color_stats(image)`按亮度分位数把像素切成阴影/中间调/高光三段分别算平均RGB，再算整体冷暖倾向、绿品倾向、饱和度、对比度。**参考图这条路径不让模型"目测"色温**——先用算法量出客观数字，模型只负责把数字翻译成风格化参数，结果比纯多模态判断稳定得多
- **`llm.py`**：OpenAI兼容客户端，走JSON输出模式。只实现这一套协议是因为它能覆盖OpenAI、DeepSeek、Moonshot、通义千问、智谱以及本地ollama/vLLM——都提供兼容端点，用户只需改`base_url`
- **`prompts.py`**：system prompt里逐条说明每个参数的含义和取值习惯（提示模型大多数风格只需要温和的数值，别把每个参数都推到极限）

### API Key的处理

Key由调用方每次传入（CLI参数或API请求体），**服务器不存盘、不进日志、不写进`JobStore`**，只在单次请求的生命周期内存在。`lut_routes.py`里模型调用失败时只回报异常类型、不回显请求内容——错误消息是最容易泄露Key的地方，有一个专门的测试用例（`test_generate_error_response_does_not_leak_the_api_key`）盯着这条。

## 项目结构

```
proimages/
  core/
    system/io.py     解码RAW（.dng/.cr2/.cr3/.nef/.arw/.raf/.rw2，用rawpy，惰性import、
                      缺失时报错指向heavy组）和JPEG/PNG/HEIC（用Pillow），
                      统一转成[0,1]浮点RGB numpy数组；save_image()负责编码回8位图片写盘
    denoise/          降噪模块，process(image)->image，当前直通
    hdr/               HDR模块，process(image)->image，当前直通
    depth_bokeh/      景深虚化模块，process(image)->image，当前直通
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
      params.py       GradeParams（pydantic模型），跨仓库公开契约，带schema_version
      grade.py        apply_grade(rgb, params)->rgb，参数化调色数学，
                      对(...,3)任意形状生效，烘焙LUT和调色图片共用这一份
      bake.py         bake_lut()/bake_cube_file()/bake_cube_text()，
                      identity表过apply_grade再写成.cube
      image_stats.py  extract_color_stats(image)->dict，按亮度分位数拆阴影/中间调/高光
                      分别统计，供参考图路径喂给模型；复用core/color.luminance()
      llm.py          OpenAI兼容客户端，params_from_description()/params_from_reference_image()
      prompts.py      system prompt和两条入口各自的user prompt模板
    color/            色彩空间共用工具；luminance(image)->ndarray按Rec.709权重算亮度，
                      grain.py、halation.py、lut_gen/grade.py、lut_gen/image_stats.py都复用它
    pipeline.py       process_image(image, lut_path=None)按denoise→hdr→depth_bokeh→physical_fx的顺序
                      依次调用，lut_path透传给physical_fx.process()
  gpu_config.py       detect_device()探测cuda/mps/cpu，支持传override参数强制指定；
                      torch是惰性import，未安装（基础安装）时返回"cpu"，
                      这样基础安装也能起完整API而不是在启动时崩掉
  model_download.py  ensure_model(repo_id)用huggingface_hub.snapshot_download把预训练权重下载到
                      ~/.cache/proimages/models/（可用PROIMAGES_MODELS_DIR环境变量改路径），已存在则直接复用缓存
  cli_args.py         add_device_argument()定义共用的--device参数，CLI和API服务入口都调用它
  cli.py              proimages命令行入口：读入一张图片路径→跑process_image()→写到输出路径
  lut_cli.py          proimages-lut命令行入口：--describe/--reference/--params三选一，
                      生成或烘焙出.cube；--params路径不调LLM也不需要Key
  api/
    app.py            FastAPI应用实例，挂载GET /health和jobs路由
    lifespan.py       FastAPI启动时用detect_device()探测设备存到app.state.device，
                      并创建一个JobStore实例存到app.state.job_store
    server_cli.py     proimages-api命令行入口：解析--host/--port/--device，用uvicorn跑api.app:app
    jobs/
      models.py       JobStatus枚举（pending/running/completed/failed）和JobRecord数据类
                      （job_id、状态、创建时间、结果字节、错误信息）
      store.py        JobStore：内存态的{job_id: JobRecord}字典，所有读写都加asyncio.Lock
      runtime.py      run_job()：把上传的图片字节解码成numpy数组→跑process_image()→编码成PNG字节
                      →写回JobStore，异常会被捕获并记成failed状态而不是让请求本身报错
    http/
      routes.py       三个接口：POST /v1/jobs（上传文件，创建任务并用asyncio.create_task后台跑，
                      立即返回job_id）、GET /v1/jobs/{job_id}（查状态，不存在返回404）、
                      GET /v1/jobs/{job_id}/result（状态不是completed时返回409，是的话返回PNG字节）
      lut_routes.py   两个同步接口：POST /v1/luts/generate（传describe或reference + model/api_key
                      /base_url，返回GradeParams和烘焙好的.cube）、POST /v1/luts/bake
                      （只传GradeParams，不调LLM不需要Key，毫秒级——这是调色盘UI拖完盘后调的那个）
```

LUT这两个接口是同步的，不走上面的job异步机制：返回体是小JSON、没有GPU排队问题、模型输出几十个数字通常2-5秒就回来，调用方在按钮上转个圈即可。`JobStore`/`JobRecord`保持只服务图像处理任务。

CLI和API服务共用同一份`core/`处理逻辑，不存在"CLI版本"和"API版本"两套算法代码。

## 依赖与运行

用[uv](https://github.com/astral-sh/uv)管理依赖。依赖分成两层，因为LUT这条链路完全用不到深度学习相关的重依赖：

| | 包含 | 装出来的虚拟环境 | 能用什么 |
|---|---|---|---|
| **基础安装** | fastapi、uvicorn、python-multipart、numpy、pillow、colour-science、scipy、openai、pydantic | ~220MB | LUT全链路：`core/lut_gen`生成LUT、`core/physical_fx`全部5个效果、完整API（含`/v1/luts/*`和`/v1/jobs`）、三个CLI入口 |
| **`heavy`可选组** | 额外加torch、rawpy、huggingface_hub | ~800MB+ | 再加上RAW格式输入，以及`core/denoise`、`core/hdr`、`core/depth_bokeh`（这三个模块目前还是占位直通，等接入预训练模型后才真正需要） |

```bash
# 只用LUT链路（比如被其他项目当依赖装）
uv sync --extra dev

# 完整安装（需要RAW输入或后续的降噪/HDR/景深模块）
uv sync --extra heavy --extra dev
```

被其他项目作为可编辑依赖引入时，不带extra即可，不会拖进torch：

```bash
uv pip install -e ../ProImages
```

基础安装下`gpu_config.detect_device()`返回`"cpu"`——torch本身就是提供GPU访问的东西，没有它就确实只有CPU，而需要GPU的那三个模块在基础安装下本来也不可用。对RAW文件调用`load_image()`会抛出明确指向`proimages[heavy]`的ImportError，而不是一个裸的`No module named 'rawpy'`。

```bash

uv run proimages input.jpg output.png --device cpu                       # CLI：处理单张照片，不调色
uv run proimages input.jpg output.png --lut "Example-LUTs/Arabica 12.CUBE"  # CLI：额外套用一个LUT

uv run proimages-api --host 127.0.0.1 --port 8001     # API：起服务
# 另开一个终端：
curl -X POST http://127.0.0.1:8001/v1/jobs -F "file=@input.jpg"                                    # 提交，不调色
curl -X POST http://127.0.0.1:8001/v1/jobs -F "file=@input.jpg" -F "lut=@Example-LUTs/Arabica 12.CUBE"  # 提交，套用LUT
curl http://127.0.0.1:8001/v1/jobs/<job_id>                        # 查状态
curl http://127.0.0.1:8001/v1/jobs/<job_id>/result -o output.png   # 完成后取结果

uv run pytest                    # 跑测试
```

生成LUT（`proimages-lut`）：

```bash
# 用一句话描述风格生成
uv run proimages-lut --describe "温暖的夏日胶片感" --model gpt-4o --api-key sk-xxx -o summer.cube

# 用参考图生成（先算色彩统计再连图给模型）
uv run proimages-lut --reference ref.jpg --model gpt-4o --api-key sk-xxx -o ref_look.cube

# 直接从参数文件烘焙，不调LLM、不需要Key
uv run proimages-lut --params params.json -o custom.cube

# 指向其他OpenAI兼容厂商
uv run proimages-lut --describe "赛博朋克冷调" --model deepseek-chat \
  --api-key sk-xxx --base-url https://api.deepseek.com/v1 -o cyber.cube

# 加 --save-params 会把AI给的参数也存成JSON，方便手动微调后重新烘焙
```

API Key也可以通过`PROIMAGES_API_KEY`环境变量传，省得每次敲。生成出来的`.cube`直接就能给上面的`proimages --lut`用。

`--device`不传时会自动探测：有CUDA用cuda，苹果芯片用mps，否则用cpu。

测试用的样张（`Example-imgs/`）和LUT文件（`Example-LUTs/`，35个`.CUBE`文件）只放在本地，`.gitignore`里已排除，不会进版本库。

## 版本记录

| 版本 | 日期 | 变更内容 | 类型 |
|------|------|----------|------|
| P2d1 | 2026-07-22 | 拆分依赖：torch/rawpy/huggingface_hub移入heavy可选组，基础安装只保留LUT链路所需（虚拟环境800M+降至~220MB）；gpu_config的torch改为惰性import并在缺失时回退cpu，使基础安装也能起完整API；RAW读取缺rawpy时报错指向heavy组 | refactor |
| P2d | 2026-07-22 | 新增AI生成LUT模块core/lut_gen：GradeParams跨仓库契约、参数化调色数学、.cube烘焙、参考图色彩统计、OpenAI兼容客户端；新增proimages-lut命令行入口和/v1/luts/{generate,bake}两个接口 | feat |
| P2c | 2026-07-21 | 实现halation（高光提取+大半径模糊+暖色调叠加）、高光滚降（smoothstep S曲线）、镜头暗角（cos⁴渐晕）三个效果，luminance计算提取到core/color供grain/halation复用；至此physical_fx的5个效果全部实现完毕 | feat |
| P2b | 2026-07-21 | 实现LUT应用（colour-science三线性插值.cube）和物理颗粒（中间调加权+空间相关噪声）两个效果，CLI加--lut参数，API的/v1/jobs加lut上传字段 | feat |
| P2a | 2026-07-21 | core/lut_grain重构为core/physical_fx，内部拆成LUT/颗粒/halation/滚降/暗角5个独立效果文件+STAGES有序效果链，便于后续新增物理效果 | refactor |
| P2 | 2026-07-21 | 完成工程骨架：FastAPI异步任务API + CLI，两者共用core/管线（4个模块均为占位直通），gpu_config设备探测，model_download权重下载占位 | milestone |
| P1 | 2026-07-21 | 初始化项目仓库与README | milestone |
