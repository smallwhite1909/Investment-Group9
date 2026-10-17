# Investment Group 9

投资学第9组：A股截面选股，比较基本面打分、综合打分、Ridge和浅层LightGBM。采用日频输入、未来20个交易日收益预测、每20个交易日调仓的等权Top100组合，数据覆盖2018-01-02至2026-04-29。

## 提交文件

| 文件 | 内容 |
| --- | --- |
| [Report_Group9.pdf](Report_Group9.pdf) | 英文报告，正文12页＋参考文献1页 |
| [Presentation_Group9.pptx](Presentation_Group9.pptx) | 13页幻灯片，讲解18分钟，Q&A另算 |
| [Presentation_Group9.pdf](Presentation_Group9.pdf) | 与PPTX同版本的预览 |
| [code/](code/)、[config.json](config.json)、[data_dictionary.md](data_dictionary.md) | 固定方案的计算代码、参数和外部输入说明 |

老师另要求单个`SupplementaryMaterials_Group9.zip`。本仓库按要求只保留13个代码、说明及展示文件；数据、模型权重、回测输出和压缩包不上传。报告和PPT可直接查看，计算需要另备下述外部输入。Canvas提交需由组员完成。

## 策略与结果

每日收盘更新分数，在连续交易日历上每20日更新目标，下一开盘交易；受限订单可逐日重试。四方法和主基准使用相同的历史完整输入股票池。主基准为该池等权组合，采用相同调仓频率、交易限制及匹配费用；CSI500价格指数仅作辅助参考。

采用年度模型和已固定的逐年因子、方向及参数，见[config.json](config.json)。先按行业和对数市值处理分数，再作因果EMA：基本面和Ridge为20，综合和LightGBM为40。Ridge的alpha为10000；LightGBM深度3、7叶子、56轮。程序直接重训固定方案，不重新搜索参数。

| 区间 | Ridge费前 | Ridge费后 | 基准费后 | 累计每日费后超额 |
| --- | ---: | ---: | ---: | ---: |
| 2021–2025 | 225.99% | 195.66% | 76.60% | +52.02个百分点 |
| 2026截至4月29日 | 9.40% | 8.80% | 8.18% | +0.56个百分点 |

佣金单边3bp、滑点5bp；卖出印花税在2023-08-28前为10bp，此后5bp。绝对收益复利累计；超额曲线累加每日策略收益减同费用基准收益，不等于两个累计收益直接相减。完整历史包含2018–2020年的拟合回放；2021年起逐年训练只使用预测开始前已成熟的标签。

## 运行

使用Python 3.9、Conda环境`ashare`，从仓库根目录执行：

```bash
conda activate ashare
python -m pip install -r requirements.txt
python code/run.py \
  --data-dir /path/to/prepared-market-arrays \
  --features /path/to/prepared-factors.npy \
  --actions /path/to/corporate-actions.csv \
  --output /data/share/XingyuLargeFiles/Investment-Group9
```

`--data-dir`提供准备好的市场数组、历史基础股票池和收益标签；`--features`为16列因子面板；`--actions`为带来源的公司事件表。具体文件、轴顺序、字段、公式见[data_dictionary.md](data_dictionary.md)。这些是输入数据，不需要任何旧预测、旧模型、研究结果或实验日志。仅克隆仓库不会取得数据；程序不包含从供应商数据库下载或重建原始财报快照的流程。

输出目录应位于仓库及home目录之外。程序在外部目录保存重训分数、连续账本、指标、原始信号基线和日频对照，以及费前/费后、绝对/超额的完整历史四图。PDF和PPT为随仓库提供的定稿文件，运行代码不自动重新排版它们。

## 代码

| 文件 | 职责 |
| --- | --- |
| `code/run.py` | 统一运行入口 |
| `code/data.py` | 加载外部数组和公司事件 |
| `code/factors.py` | 因子定义及共同股票池 |
| `code/models.py` | 年度训练、截面处理及因果平滑 |
| `code/backtest.py` | 成交、费用、公司事件、指标和四图 |

价格链近似再投资分红；财报历史修订无法完全排除。回测省略整手取整、最低佣金和流动性冲击，历史收益不保证未来表现。
