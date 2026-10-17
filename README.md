# Investment Group 9 · V2

投资学第9组：A股截面选股，比较基本面打分、综合打分、Ridge和浅层LightGBM。采用日频输入、未来20个交易日收益预测、每20个交易日调仓的等权Top100组合，数据覆盖2018-01-02至2026-04-29。

## 提交文件

| 文件 | 内容 |
| --- | --- |
| [Report_Group9.pdf](Report_Group9.pdf) | 英文报告，正文14页＋参考文献1页，含完整历史四图和诊断分析 |
| [Presentation_Group9.pptx](Presentation_Group9.pptx) | 15页16:9幻灯片，文字与表格可编辑，备注合计18分钟，Q&A另算 |
| [Presentation_Group9.pdf](Presentation_Group9.pdf) | 与PPTX使用同一内容、几何布局及图表生成的PDF预览 |
| [code/](code/)、[config.json](config.json)、[data_dictionary.md](data_dictionary.md) | 固定方案的计算代码、参数和外部输入说明 |

老师另要求单个`SupplementaryMaterials_Group9.zip`。本仓库保留14个必要文件；V2新增一个集中生成图表、报告和PPT的模块。数据、模型权重、回测输出和压缩包不上传。报告和PPT可直接查看，计算需要另备下述外部输入。Canvas提交、组员分工和彩排需由组员完成。

V2补充回撤、月度热力图、年度收益、滚动超额与波动、换手和费用、因子相关性、Rank IC、五分组及实际持仓市值分布。费前/费后绝对与超额四图保留完整历史，沿用原模型和收益账本。

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

`--data-dir`提供准备好的市场数组、历史基础股票池和收益标签；`--features`为16列因子面板；`--actions`为带来源的公司事件表。具体文件、轴顺序、字段、公式见[data_dictionary.md](data_dictionary.md)。完整运行从这些输入重新训练，不依赖旧项目的研究工件。仅克隆仓库不会取得数据；程序不包含从供应商数据库下载或重建原始财报快照的流程。

默认`--stage all`会重训固定模型、重建回测并生成报告和展示。已有本管线的`models/`、`results/`和`raw_baseline/`输出后，在同一命令中加`--stage report`可只重算诊断和重建文档；仍需原外部数据，用于信号诊断及实际持仓核验。

输出目录应位于仓库及home目录之外。重训分数、连续账本及对照分别位于`models/`、`results/`和`raw_baseline/`；图表、数值表和生成核验记录位于`report_assets/`。三份正式文档更新到仓库根目录。PPT文字、数字和表格为原生对象，图表为单独插入的高清图片；PDF保留矢量图表，按相同元素布局生成，未经PowerPoint原生导出验证。

Rank IC和五分组是未来20日标签诊断，不是费后交易收益；2021–2025诊断只用截至2025年底已成熟的标签，2026诊断截止4月29日，最后可评估的信号为3月30日。实际持仓图来自净值账本重放后的收盘权重，已核对重放交易数值不变。日频双倍滑点仍是日频对照。

## 代码

| 文件 | 职责 |
| --- | --- |
| `code/run.py` | 统一运行入口 |
| `code/data.py` | 加载外部数组和公司事件 |
| `code/factors.py` | 因子定义及共同股票池 |
| `code/models.py` | 年度训练、截面处理及因果平滑 |
| `code/backtest.py` | 成交、费用、公司事件、指标和可选只读持仓快照 |
| `code/report.py` | 固定结果诊断、全部图表、英文报告、可编辑PPT及PDF |

价格链近似再投资分红；财报历史修订无法完全排除。回测省略整手取整、最低佣金和流动性冲击，历史收益不保证未来表现。
