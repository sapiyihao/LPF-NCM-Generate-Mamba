# dataset2 长序列缓存与故障诊断

本仓库保存可复现实验的代码与轻量结果，不包含原始 `dataset2` CSV 或生成后的大型缓存。先安装 `requirements.txt`，将原始数据放在本机可访问的位置，再运行：

```powershell
python -m pip install -r requirements.txt
python prepare_dataset.py --source "D:\path\to\dataset2" --output data/processed_v2
python inspect_dataset.py --cache data/processed_v2
python train.py --cache data/processed_v2 --chemistry LFP
```

`prepare_dataset.py` 一次性建库，后续训练只读缓存，不需要反复预处理。同名输出目录已存在时脚本会拒绝覆盖。下文的 `H:\mamba\...` 是本次实验机器上的示例路径，其他机器请替换为自己的路径。

本工程独立于第一套电池包特征数据。`dataset2` 的标签来自车辆目录，共四类：正常、低容量、高内阻、自放电；属于**车辆级标签**，不是故障发生时刻或故障单体标签。

## 一次性生成

```powershell
& 'D:\enviroment\anaconda\envs\py312\python.exe' inspect_dataset.py --cache H:\mamba\code2\data\processed_v2
```

全量缓存已生成在 `data/processed_v2`。所有训练、验证和测试直接使用该目录，无需重新读取原始CSV。原始的 `processed_v1` 经审计发现跨VIN重复序列，已经标记失效，不可用于正式实验。若以后创建全新版本，可运行 `prepare_dataset.py --source H:\mamba\dataset2 --output H:\mamba\code2\data\processed_v3`；脚本会在建库阶段自动把完全重复序列对应的VIN分到同一集合，且拒绝覆盖已存在的目录。

缓存包含：

- `series.f32`：按原CSV时间顺序连续存放的7通道浮点数据；
- `index.npz`：序列边界、车辆划分和窗口索引；
- `normalization.npz`：按LFP/NCM分别计算、只来自训练车辆的统计量；
- `vehicles.json`、`series.jsonl`：可追溯的车辆和原文件信息；
- `manifest.json`：构建参数、样本统计及源文件清单指纹。

默认窗口最长512个真实时间点（完整窗口约85分钟），步长256。短CSV右侧补零并提供有效位 `mask`，补零**不代表**真实时间点。窗口**不会跨CSV**。一个训练输入 `x` 为 `[B,512,7]`，7个通道依次为平均单体电压、最高/最低单体电压、总电流、最高/最低温度、单体电压极差。平均单体电压用实际 `VOLT_*` 列数换算，避免不同串数造成不合理电压。当前缓存不纳入 `SOC` 与单体个数作为输入。

数据先按化学体系和类别分层、再以 `chemistry/VIN` 为最小单位拆成70%/15%/15%。所有CSV和窗口继承车辆的划分，绝不在窗口层随机分割。LFP和NCM共用缓存格式，但标准化参数分别拟合；建模时建议分别训练和报告，避免化学体系混杂。

```python
from longseq.data import LongSequenceDataset

train = LongSequenceDataset('data/processed_v2', split='train', chemistry='LFP')
item = train[0]
print(item['x'].shape, item['mask'].shape, item['label'])
```

注意：高度重叠的窗口不能当成相互独立的测试对象。最终指标应按车辆聚合，并与“只看电压/电流/温度”的非Mamba基线比较。类别分布和有效窗口长度见 `manifest.json`；若任一类别系统性被跳过，须先解决数据质量问题再训练。

**长度混杂风险：** LFP仅约3%的窗口短于512；NCM中正常、高内阻、自放电类别各约59%短于512，而低容量约16%。NCM模型可能凭有效长度而非电学特征分类。因此建议先以LFP作为长序列主实验；NCM必须做统一真实长度、长度匹配抽样或长度基线等额外控制后再解释指标。

## 长序列网络入口

```powershell
# 先跑端到端冒烟测试，不使用全量数据、指标没有统计意义
& 'D:\enviroment\anaconda\envs\py312\python.exe' train.py --cache H:\mamba\code2\data\processed_v2 --chemistry LFP --smoke-test

# 正式实验：LFP和NCM分别训练
& 'D:\enviroment\anaconda\envs\py312\python.exe' train.py --chemistry LFP
& 'D:\enviroment\anaconda\envs\py312\python.exe' train.py --chemistry NCM
```

模型以512个**真实时间点**作为序列：7维输入投影、两层输入相关的选择性状态空间块、掩码池化、四分类头。训练窗口按“类别→车辆→窗口”加权抽样，减少正常类和长CSV的支配；验证和测试按车辆聚合窗口概率，并以验证车辆Macro-F1保存最佳权重。正式训练默认batch=64，实验分别保存到带时间戳的 `experiments/` 目录。这里使用的是可在当前Windows/PyTorch环境直接运行的 **Mamba风格纯PyTorch选择性SSM**，不是官方 `mamba-ssm` fused CUDA实现；论文中不能把两者的速度或架构视为完全相同。VS Code请直接打开 `H:\mamba\code2`，再选择 `.vscode/launch.json` 中的启动项。

## 首轮正式实验

`experiments/20260919_135510_LFP_long_mamba/` 保存了 LFP 实验的配置、逐轮指标、最佳权重、结果图和摘要。随机种子42，12轮，验证车辆Macro-F1最佳为第8轮的0.893；独立测试75台车，准确率0.973、Macro-F1为0.954，其中15台故障车全部分对，2台正常车被误判为低容量。故障测试车辆每类仅4–6台，此成绩只是单次划分结果，需要多种子/多划分验证，不能当作稳健泛化结论。其他冒烟测试及未来实验默认由 `.gitignore` 排除，正式记录须明确选择后再纳入版本控制。
