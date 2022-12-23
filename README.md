

##  安装
  - pip install -U deep_training >= 0.0.3
  - 当前文档版本pypi 0.0.3

## 更新详情
  - [deep_training](https://github.com/ssbuild/deep_training)

## 目录
  - task_classify 分类模型
  - task_extract_ner 序列抽取模型
  - tast_extract_relation 关系抽取模型
  - task_generate 文本生成模型
  - task_pretrain 主流预训练模型
  - task_sentense_embeddings 句向量模型
  - task_custom_muti_gpu 更多自定义训练操作，例如多卡训练例子， 模型转换onnx 等一些列自定义操作
## 多卡训练策略 strategy
    # Available names: bagua, colossalai, ddp, ddp_find_unused_parameters_false, ddp_fork,
    # ddp_fork_find_unused_parameters_false, ddp_fully_sharded,
    # ddp_notebook, ddp_notebook_find_unused_parameters_false, ddp_sharded,
    # ddp_sharded_find_unused_parameters_false, ddp_sharded_spawn,
    # ddp_sharded_spawn_find_unused_parameters_false,
    # ddp_spawn, ddp_spawn_find_unused_parameters_false,
    # deepspeed, deepspeed_stage_1, deepspeed_stage_2, deepspeed_stage_2_offload,
    # deepspeed_stage_3, deepspeed_stage_3_offload, deepspeed_stage_3_offload_nvme,
    # dp, fsdp, fsdp_native, fsdp_native_full_shard_offload, horovod, hpu_parallel,
    # hpu_single, ipu_strategy, single_device, single_tpu, tpu_spawn, tpu_spawn_debug"
## 愿景
创建一个模型工厂, 轻量且高效的训练程序，让训练模型更容易,更轻松上手。

## 交流
QQ交流群：185144988
