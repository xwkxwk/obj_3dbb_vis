# obj_3dbb_vis

`obj_3dbb_vis` 将 `obj-centric-3dbb@dc5ee99` 的可视化流水线接入
`sim-infrastructure==2.3` 的 `Realtime_Processor_Engine`。HTTP、初始化、启动、
停止和 Redis 调度均由基类负责，适配类只实现处理钩子。服务逐帧读取 RGB-D
和相机位姿，只向场景目录发布 2D 图、3D 图和增量 Fuse CSV，不向 Redis
写回结果。

## HTTP 接口

服务监听 `8883`。health、init、start 和 stop 接口及其响应格式均来自
`Realtime_Processor_Engine` 基类，例如：

```json
{
  "status": "ok",
  "code": 200,
  "data": {}
}
```

- `/api/health`：使用基类健康检查。
- `/api/realtime/init`：使用基类初始化，只重置帧号和 `stop_flag`。
- `/api/realtime/recon3d-params`：设置本次任务的外部相机内参。
- `/api/realtime/start`：由基类启动 Redis producer/consumer。
- `/api/realtime/stop`：取消当前场景输出并等待自有 worker 退出，模型和服务保持常驻。

相机参数设置后保留到进程退出，基类 init 不会清除它们。

### HTTP 接口调用顺序

一次完整任务按以下顺序调用：

```text
启动服务进程
  -> 等待 POST /api/health 成功               # 模型加载完成且 HTTP 已启动
  -> POST /api/realtime/init                 # 重置基类帧号和停止标志
  -> POST /api/realtime/recon3d-params       # 设置相机内参
  -> POST /api/realtime/start                # 启动基类 Redis 读取线程
  -> 持续向 Redis Stream 写入 RGB-D 和位姿
  -> POST /api/realtime/stop                 # 停止当前场景，服务继续运行
```

进程日志中的 `init_processor` 是模型初始化，不是 HTTP `/api/realtime/init`。
基类在模型加载完成后才启动 HTTP，因此调用方必须等待 health 成功，再调用一次
`init`。`init`、`recon3d-params` 和 `start` 必须在开始写入本次任务数据前完成。
不要在任务运行中再次调用 `init` 或 `start`。场景切换必须按 `stop → init →
可选更新相机参数 → start` 执行；BoxerNet 和 GroundingDINO 在同一服务进程中
保持常驻，不会因场景 stop 重新加载。

以下命令给出一套完整示例：

```bash
# 1. 等待模型加载完成且 HTTP 服务可用
until curl -fsS -X POST http://127.0.0.1:8883/api/health \
  -H 'Content-Type: application/json' -d '{}'; do
  sleep 1
done

# 2. 初始化基类状态
curl -X POST http://127.0.0.1:8883/api/realtime/init \
  -H 'Content-Type: application/json' \
  -d '{}'

# 3. 设置相机参数
curl -X POST http://127.0.0.1:8883/api/realtime/recon3d-params \
  -H 'Content-Type: application/json' \
  -d '{
    "topic_name": "simu4d:perceive:camera",
    "intrinsics": {
      "intrinsics": [612.117102, 0, 639.518311, 0, 613.186547, 359.725067900, 0, 0, 1],
      "width": 1280,
      "height": 720
    }
  }'

# 4. 启动任务；随后基类线程持续读取 input_topic_name 对应的 Redis Stream
curl -X POST http://127.0.0.1:8883/api/realtime/start \
  -H 'Content-Type: application/json' \
  -d '{
    "scene_id": "scene001",
    "start_timestamp": 0,
    "input_topic_name": "simu4d:perceive:camera"
  }'

# 5. 停止当前场景；接口返回后服务和模型继续常驻
curl -X POST http://127.0.0.1:8883/api/realtime/stop \
  -H 'Content-Type: application/json' \
  -d '{}'
```

`output_topic_name` 可兼容传入 start 请求，但不会使用。实际 Redis Stream 使用
start 请求中的 `input_topic_name`；YAML 中的 `input_queue_name` 只是调用方默认
约定。核心接口的状态码和异常响应以 `sim-infrastructure==2.3` 为准。

### Engine 类内部调用顺序

服务入口及模型初始化由基类生命周期驱动：

```text
main()
  -> init()                                      # 创建 Redis 客户端并读取配置
  -> Obj3DBBVis_Realtime_Engine(...)
       -> Realtime_Processor_Engine.__init__()   # 创建 Flask 并注册基类接口
  -> engine.load_processor()                     # 基类方法
       -> engine.init_processor()                # 子类钩子
            -> 注册 recon3d-params
            -> 创建 RealtimeVisProcessor
            -> 加载 BoxerNet、启动 GroundingDINO
  -> engine.run()                                # 基类方法
       -> engine.thread_api()                    # Flask 后台线程
```

收到 start 请求后的调用链为：

```text
基类 start_predict()
  -> 设置 scene_path
  -> 基类 do_start()
       -> 子类 pre_start_thread()
            -> RealtimeVisProcessor.pre_start()
            -> 创建输出目录、帧 worker 和 Fuse worker
       -> 基类 thread_calculate_producer()       # Redis 读取线程
       -> 基类 thread_calculate_consumer()       # 结果消费线程，本服务无 Redis 输出
```

每条 Redis 消息的处理顺序为：

```text
thread_calculate_producer()
  -> Redis_Client.xread_from_id()
  -> handle_inputdata()                          # 提取时间戳和原始 RGB-D/位姿
  -> do_processor_function()
       -> RealtimeVisProcessor.submit_frame()    # 写入单个最新帧槽并立即返回
  -> RealtimeVisProcessor._frame_loop()          # 独立后台线程
       -> decode_frame()
       -> process_frame()
       -> 发布 2D/3D 图片
       -> IncrementalFuseWorker.submit()         # 有最终 3DBB 时提交 Fuse
```

收到 stop 请求后的调用链为：

```text
基类 stop_predict()
  -> 基类 do_stop()                              # 先设置 stop_flag，停止 Redis 收帧
       -> 子类 stop_processor()
            -> RealtimeVisProcessor.close()
            -> 封锁当前场景的图片和 CSV 发布
            -> 丢弃等待帧和等待 Fuse 批次
            -> 等待自有 frame/Fuse worker 自然退出
            -> 保留 BoxerNet、GroundingDINO、Redis 和 HTTP 服务
```

`close()` 不设置等待超时，也不等待基类 Redis producer/consumer。若 DINO、Boxer
或 Fuse 永久卡住，stop 接口也会一直阻塞，只能从容器外强制终止。该行为与
Splatam 只等待自身处理线程的语义一致。由于基类没有保存和 join Redis 线程，
快速执行下一次 start 存在旧 producer/consumer 尚未退出的竞态；本服务明确沿用
该基类行为。

### 不通过 HTTP 直接调用基类

HTTP 只是 `do_start()` 和 `do_stop()` 的控制入口。如果同一进程中的 Python
模块已经持有 Engine 实例，也可以直接调用基类方法。调用方必须完成 HTTP 路由
原本负责的相机参数和 `scene_path` 设置：

```python
import time

from obj3dbb_vis_engine import Obj3DBBVis_Realtime_Engine, init

redis_client, processor_config, full_config, config_path = init()
engine = Obj3DBBVis_Realtime_Engine(
    redis_client,
    processor_config,
    full_config,
    config_path,
)
engine.load_processor()

scene_id = "scene001"
topic_name = "simu4d:perceive:camera"
engine.rgb_topic = topic_name
engine.intrinsics = [900, 0, 640, 0, 900, 360, 0, 0, 1]
engine.width = 1280
engine.height = 720
engine.scene_path = engine.base_scene_path / scene_id

engine.do_start(scene_id, topic_name, 0, None)
try:
    while not engine.stop_flag:
        time.sleep(1)
finally:
    engine.do_stop()
```

这种方式不调用 `engine.run()`，因此不会启动 Flask HTTP 线程。`do_start()` 仍由
基类启动相同的 Redis producer/consumer；`do_stop()` 仍进入相同的安全收尾逻辑，
但不会退出服务进程。示例主程序结束后由操作系统释放常驻模型。标准部署仍推荐
使用前述 HTTP 调用顺序。

## Redis 输入

每条 Stream 消息包含：

- `ts`：纳秒时间戳，直接用作输出文件名。
- `content`：JPEG RGB 字节。
- `depth`：`uint16` PNG 深度字节，单位为毫米。
- `camera_pose`：pickle 字典，包含 `position.{x,y,z}` 和
  `orientation.{x,y,z,w}`。

服务端不根据 `ts` 判断帧的先后顺序，而是按 Redis Stream 返回的消息顺序提交。
时间戳顺序和唯一性由上游保证；相同 `ts` 会对应相同输出文件名。缺字段、时间戳
格式错误、解码失败、尺寸不匹配或位姿无效的帧会被记录并跳过。
GroundingDINO、Boxer、Fuse 或文件发布异常会停止处理器和基类 Redis 循环，
并写入服务日志。基类 health 不额外暴露算法错误状态。

基类 Redis 线程只负责提取并提交帧。处理器内部只保存一个尚未开始处理的最新帧；
新帧可覆盖该槽位，但不会中断已经开始的推理。

### 离线 RGB-D 数据回放

宿主机的普通 Python 环境不一定安装了 `redis-py`。推荐在服务容器启动、相机参数
设置且 start 接口调用成功后，使用容器内的 Boxer Conda 环境执行回放脚本：

```bash
docker exec -it proc-sim-obj3dbb-vis \
  /opt/conda/envs/xwk_boxer/bin/python \
  scripts/replay_rgbd_to_redis.py \
  --data /data/data260814 \
  --stream simu4d:perceive:camera \
  --fps 1 \
  --clear-stream
```

这里的项目根目录 `data/data260814` 在容器中对应 `/data/data260814`。回放脚本
要求该目录包含 `images/`、`depth/` 和 `poses.txt`。脚本不排序或校验业务
时间戳，而是按 `poses.txt` 中的原始顺序，将存在对应 RGB 和深度文件的帧依次
写入 Redis；Redis 自动 Stream ID 决定消息顺序。服务端只将 `ts` 解析为非负
整数并用于文件名，不检查时间戳是否递增。
`--clear-stream` 会先删除指定流，只应在确认该流用于当前测试时使用。

## 输出约定

每次 start 只清空并重建当前场景的三个结果子目录：

```text
/data/scene/<scene_id>/obj3dbb_vis/
  2D_detections/<ts>.jpg
  3D_detections/<ts>.jpg
  fused/<截止帧ts>.csv
```

两张图片使用成对原子发布，JPEG 质量为 85；2D 图绘制过滤前的原始 DINO
框，3D 图只绘制最终 validation 后的 3DBB。Fuse CSV 使用 SFM 坐标、稳定
`fused_instance`、中文语义和词表 RGB 色值。没有最终 3DBB 时不会提交 Fuse。
stop 会先封锁当前场景的最终文件发布，再等待正在计算的帧和 Fuse 自然返回；
这些计算生成的临时文件会被删除，不会成为新的 JPG 或 CSV。stop 生效前已经
完成原子发布的历史结果会保留。

实时读取方应取 `2D_detections/` 与 `3D_detections/` 文件名交集中的最大数值
时间戳；Fuse 读取方取 `fused/` 中最大数值时间戳。服务不会通过 Redis 发布路径，
也不生成 timing 或显存统计文件。

## GPU 运行模式

GPU 模式由 `conf/config.yaml` 设置，默认使用两张卡：

```yaml
gpu:
  mode: dual
  detection_device: cuda:0
  fuse_device: cuda:1
```

- `dual`：GroundingDINO、Boxer 和逐帧检测使用 `detection_device`，增量 Fuse
  使用不同的 `fuse_device`。
- `single`：检测与 Fuse 都使用 `detection_device`，配置中的 `fuse_device` 被忽略。

设备名必须使用明确的 `cuda:N` 格式。CUDA 不可用、编号超出容器内可见 GPU
范围，或双卡模式把检测和 Fuse 配置到同一设备时，服务会在模型初始化阶段直接
失败，不会降级到单卡或 CPU。GPU 配置只在服务启动时读取；修改运行模式或设备
编号后必须重启服务。容器继续通过 `--gpus all` 暴露全部 GPU，配置中的编号是
容器内可见设备编号。

单卡配置示例：

```yaml
gpu:
  mode: single
  detection_device: cuda:0
  fuse_device: cuda:1  # 单卡模式下忽略
```


## 服务部署

启动脚本会根据 `scripts/start.sh` 自身位置解析项目根目录，不依赖固定的宿主机
绝对路径。默认目录结构为：

```text
obj_3dbb_vis/
  data/
  scripts/
  src/
```

默认将项目根目录中的 `data/` 挂载为容器内的 `/data`。如果服务器上的场景数据
不在该位置，通过 `OBJ3DBB_DATA_HOME` 指定宿主机目录。

将模型文件放在该目录的 `ckpts/`：

```text
ckpts/
  boxernet_hw960in4x6d768-wssxpf9p.ckpt
  groundingdino_swinb_cogcoor.pth
  bert-base-uncased/
```

启动脚本会将服务的 `src/`、`conf/`、`scripts/`、`libs/boxer/` 覆盖挂载到基础镜像的
`/workspace/boxer`，`ckpts/` 只读挂载，`logs/` 可写挂载，场景数据挂载到
`/data`。容器使用全部 GPU、host 网络，名称为
`proc-sim-obj3dbb-vis`。

```bash
cd /data4/xwk/simulator-service/obj_3dbb_vis
bash scripts/start.sh
```

使用自定义场景数据目录：

```bash
OBJ3DBB_DATA_HOME=/data4/xwk/my-simulator-data bash scripts/start.sh
```

调试模式只进入容器交互 shell，不自动启动服务：

```bash
bash scripts/start.sh --mode debug
```

普通 `/api/realtime/stop` 只停止当前场景，不退出容器。停止脚本先调用该接口并
等待场景 worker 结束，再显式停止容器；场景 close 没有超时，若内部线程永久
卡住，脚本也会阻塞：

```bash
bash scripts/stop.sh
```

日志同时写到控制台与 `logs/`，文件每小时轮转、压缩并保留 7 天。
