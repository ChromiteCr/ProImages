# ProImages

AI + 物理建模的手机/运动相机照片"相机化"处理管线。将手机或运动相机拍摄的照片，通过景深虚化、降噪、动态范围提升（HDR）、LUT调色/胶片质感四个模块处理，使其在观感上更接近专业相机直出的效果。

工程架构参照 [ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5)：Python包 + FastAPI服务 + CLI，uv管理依赖，Docker打包，API优先设计，便于后续发布和软件包装。

## 版本记录

| 版本 | 日期 | 变更内容 | 类型 |
|------|------|----------|------|
| P2 | 2026-07-21 | 完成工程骨架：FastAPI异步任务API + CLI，两者共用core/管线（4个模块均为占位直通），gpu_config设备探测，model_download权重下载占位 | milestone |
| P1 | 2026-07-21 | 初始化项目仓库与README | milestone |
