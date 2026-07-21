# ProImages

把手机或运动相机拍的照片，通过AI模型和物理光学/胶片建模处理成更接近专业相机直出的效果。处理链路包含四个模块：

1. **降噪**（`proimages/core/denoise`）：目标是用预训练的Restormer或NAFNet处理手机小底传感器的高噪点，RAW输入会在去马赛克之前先在拜耳域做降噪
2. **HDR/动态范围**（`proimages/core/hdr`）：目标是先用经典exposure fusion算法恢复单张照片的高光/阴影细节，后续再考虑单图HDR重建的深度学习方法
3. **景深虚化**（`proimages/core/depth_bokeh`）：目标是用Depth Anything V2/MiDaS做单目深度估计，再用物理光学模型（弥散圆随景深和光圈变化、光圈形状、遮挡边缘正确合成）渲染虚化，而不是简单高斯模糊
4. **LUT调色与物理相机/胶片效果**（`proimages/core/physical_fx`）：应用行业标准`.cube` LUT文件调色，再叠加物理胶片颗粒、halation（高光光晕）、高光滚降、镜头暗角（cos⁴渐晕）等物理效果。这个模块内部是一条可扩展的"物理效果链"（`effects.py`里的`STAGES`列表），每个效果都是独立文件、独立函数，以后要加色差（chromatic aberration）、镜头光晕（lens flare）、衍射星芒等新的物理效果，只要新增一个文件并加进`STAGES`列表即可，不需要改这个模块之外的任何代码

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

## 项目结构

```
proimages/
  core/
    system/io.py     解码RAW（.dng/.cr2/.cr3/.nef/.arw/.raf/.rw2，用rawpy）和JPEG/PNG/HEIC（用Pillow），
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
    color/            色彩空间共用工具；luminance(image)->ndarray按Rec.709权重算亮度，
                      grain.py和halation.py都复用它
    pipeline.py       process_image(image, lut_path=None)按denoise→hdr→depth_bokeh→physical_fx的顺序
                      依次调用，lut_path透传给physical_fx.process()
  gpu_config.py       detect_device()探测cuda/mps/cpu，支持传override参数强制指定
  model_download.py  ensure_model(repo_id)用huggingface_hub.snapshot_download把预训练权重下载到
                      ~/.cache/proimages/models/（可用PROIMAGES_MODELS_DIR环境变量改路径），已存在则直接复用缓存
  cli_args.py         add_device_argument()定义共用的--device参数，CLI和API服务入口都调用它
  cli.py              proimages命令行入口：读入一张图片路径→跑process_image()→写到输出路径
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
```

CLI和API服务共用同一份`core/`处理逻辑，不存在"CLI版本"和"API版本"两套算法代码。

## 依赖与运行

用[uv](https://github.com/astral-sh/uv)管理依赖，`pyproject.toml`里声明了fastapi、uvicorn、numpy、pillow、rawpy、torch、huggingface_hub、colour-science、scipy等核心依赖，`dev`可选依赖组里是pytest和httpx。

```bash
uv sync --extra dev              # 安装依赖（含测试用的dev组）

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

`--device`不传时会自动探测：有CUDA用cuda，苹果芯片用mps，否则用cpu。

测试用的样张（`Example-imgs/`）和LUT文件（`Example-LUTs/`，35个`.CUBE`文件）只放在本地，`.gitignore`里已排除，不会进版本库。

## 版本记录

| 版本 | 日期 | 变更内容 | 类型 |
|------|------|----------|------|
| P2c | 2026-07-21 | 实现halation（高光提取+大半径模糊+暖色调叠加）、高光滚降（smoothstep S曲线）、镜头暗角（cos⁴渐晕）三个效果，luminance计算提取到core/color供grain/halation复用；至此physical_fx的5个效果全部实现完毕 | feat |
| P2b | 2026-07-21 | 实现LUT应用（colour-science三线性插值.cube）和物理颗粒（中间调加权+空间相关噪声）两个效果，CLI加--lut参数，API的/v1/jobs加lut上传字段 | feat |
| P2a | 2026-07-21 | core/lut_grain重构为core/physical_fx，内部拆成LUT/颗粒/halation/滚降/暗角5个独立效果文件+STAGES有序效果链，便于后续新增物理效果 | refactor |
| P2 | 2026-07-21 | 完成工程骨架：FastAPI异步任务API + CLI，两者共用core/管线（4个模块均为占位直通），gpu_config设备探测，model_download权重下载占位 | milestone |
| P1 | 2026-07-21 | 初始化项目仓库与README | milestone |
