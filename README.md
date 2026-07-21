# ProImages

把手机或运动相机拍的照片，通过AI模型和物理光学/胶片建模处理成更接近专业相机直出的效果。处理链路包含四个模块：

1. **降噪**（`proimages/core/denoise`）：目标是用预训练的Restormer或NAFNet处理手机小底传感器的高噪点，RAW输入会在去马赛克之前先在拜耳域做降噪
2. **HDR/动态范围**（`proimages/core/hdr`）：目标是先用经典exposure fusion算法恢复单张照片的高光/阴影细节，后续再考虑单图HDR重建的深度学习方法
3. **景深虚化**（`proimages/core/depth_bokeh`）：目标是用Depth Anything V2/MiDaS做单目深度估计，再用物理光学模型（弥散圆随景深和光圈变化、光圈形状、遮挡边缘正确合成）渲染虚化，而不是简单高斯模糊
4. **LUT调色/胶片质感**（`proimages/core/lut_grain`）：目标是应用行业标准`.cube` LUT文件调色，再叠加物理胶片颗粒、halation（高光光晕）、高光滚降、镜头暗角（cos⁴渐晕）等物理效果

四个模块目前都是占位直通（identity），只把图片原样传递到下一步，还没有实现上述真实算法。

## 项目结构

```
proimages/
  core/
    system/io.py     解码RAW（.dng/.cr2/.cr3/.nef/.arw/.raf/.rw2，用rawpy）和JPEG/PNG/HEIC（用Pillow），
                      统一转成[0,1]浮点RGB numpy数组；save_image()负责编码回8位图片写盘
    denoise/          降噪模块，process(image)->image，当前直通
    hdr/               HDR模块，process(image)->image，当前直通
    depth_bokeh/      景深虚化模块，process(image)->image，当前直通
    lut_grain/        LUT/胶片质感模块，process(image)->image，当前直通
    color/            色彩空间共用工具，尚未写入内容，留作后续色彩转换代码的落点
    pipeline.py       process_image()按denoise→hdr→depth_bokeh→lut_grain的顺序依次调用四个模块
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

用[uv](https://github.com/astral-sh/uv)管理依赖，`pyproject.toml`里声明了fastapi、uvicorn、numpy、pillow、rawpy、torch、huggingface_hub等核心依赖，`dev`可选依赖组里是pytest和httpx。

```bash
uv sync --extra dev              # 安装依赖（含测试用的dev组）

uv run proimages input.jpg output.png --device cpu   # CLI：处理单张照片

uv run proimages-api --host 127.0.0.1 --port 8001     # API：起服务
# 另开一个终端：
curl -X POST http://127.0.0.1:8001/v1/jobs -F "file=@input.jpg"    # 提交，拿到job_id
curl http://127.0.0.1:8001/v1/jobs/<job_id>                        # 查状态
curl http://127.0.0.1:8001/v1/jobs/<job_id>/result -o output.png   # 完成后取结果

uv run pytest                    # 跑测试
```

`--device`不传时会自动探测：有CUDA用cuda，苹果芯片用mps，否则用cpu。

测试用的样张（`Example-imgs/`）和LUT文件（`Example-LUTs/`，35个`.CUBE`文件）只放在本地，`.gitignore`里已排除，不会进版本库。

## 版本记录

| 版本 | 日期 | 变更内容 | 类型 |
|------|------|----------|------|
| P2 | 2026-07-21 | 完成工程骨架：FastAPI异步任务API + CLI，两者共用core/管线（4个模块均为占位直通），gpu_config设备探测，model_download权重下载占位 | milestone |
| P1 | 2026-07-21 | 初始化项目仓库与README | milestone |
