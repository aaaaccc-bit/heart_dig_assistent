# CardioScope 心脏风险分析工作台

CardioScope 是一个基于 PySide6 的中文桌面应用，面向课程实践、统计分析演示和模型评估。它内置 303 条 Cleveland 心脏病样例数据，也支持导入符合相同字段约定的 CSV。加载数据后，可以查看人群指标、比较分类模型、探索风险分层，并将分析摘要生成可人工审核的 Markdown 报告。

![CardioScope 工作台示例，展示 Cleveland 样例的阳性率、年龄分布和分组比较](preview_workbench.png)

## 功能概览

| 页面 | 可以做什么 |
| --- | --- |
| 工作台 | 查看样本数、阳性率、年龄分布、主要风险因素及阳性组与阴性组的指标差异 |
| 样本分析 | 运行 Logistic 回归、随机森林和神经网络，比较三类模型的 ROC、PR 曲线与不同阈值下的分类指标 |
| 风险分层 | 查看模型预测概率分档、人群画像和无监督分群结果 |
| 特征洞察 | 查看特征重要性、单变量判别能力和特征相关性 |
| 分析报告 | 汇总当前数据集的分析结果，生成报告草稿，支持取消、人工修订、通过审核和导出 Markdown |
| 系统设置 | 配置报告模型、第三方 API、主题和报告安全策略 |

## 快速开始

在安装了 Git 和 Python 的 Windows 电脑上，打开 PowerShell，运行以下命令。

```powershell
git clone https://github.com/aaaaccc-bit/heart_dig_assistent.git
cd heart_dig_assistent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

在登录页使用页面底部显示的演示账号。登录后会加载内置 Cleveland 数据集，可以直接查看工作台和各分析页。AI 报告使用「系统设置」中选定的模型服务。使用 Qt Creator 时，可打开项目的 `pyproject.toml`，选用安装了项目依赖的 Python 解释器，并运行 `main.py`。

## 导入自己的数据

1. 在左侧选择「数据导入」，打开带表头、以逗号分隔的 UTF-8 CSV。
2. 导入后在工作台核对数据集名称和样本数。样本分析、风险分层、特征洞察和报告都会使用当前数据集。

CSV 需包含以下字段，列名 `num` 可作为 `target` 的别名。`target` 中 `0` 表示阴性，`1` 至 `4` 归为阳性。`ca` 和 `thal` 的缺失值可写为 `?`。

```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
```

字段的详细说明和取值范围以项目的数据约定为准。可以先用内置 CSV 对照自己的表头与编码。[数据说明](algorithm/data/README.md) · [字段约定](algorithm/schema.py) · [内置 CSV](algorithm/data/cleveland.csv)

## 配置 AI 报告

打开「系统设置」→「报告模型」。可以选择应用内置的供应商，也可以选择「自定义 API」，接入提供 OpenAI 或 Anthropic 兼容协议的服务。

| 设置项 | 自定义 API 的填写方式 |
| --- | --- |
| API 协议 | 按服务方文档选择「OpenAI 兼容」或「Anthropic 兼容」 |
| API 地址 | 填写服务方提供的基础地址，也可填写以 `/chat/completions` 或 `/messages` 结尾的完整生成地址。远程服务使用 HTTPS |
| API Key | 填写服务方签发的密钥。允许免鉴权的本机回环服务可留空 |
| 模型 | 点「获取模型」后从下拉列表选择，也可手动输入准确的模型 ID |

填完后点击「测试连接」，确认所选模型能返回短文本，再点击「保存并用于报告」。手输的 API Key 在当前应用会话内有效。自定义 API 也可在启动前通过 `CARDIOSCOPE_CUSTOM_API_KEY` 环境变量提供密钥。使用远程服务生成报告时，应用会向该服务发送当前数据集的分析摘要。

### 本机 Ollama 示例

先启动 Ollama 并下载所需模型，然后在「报告模型」选择「自定义 API」与「OpenAI 兼容」。API 地址填写 `http://127.0.0.1:11434/v1`，模型选择 Ollama 中已安装的模型。使用本机免鉴权配置时，API Key 留空。完成测试后保存设置。

### 生成与导出

1. 打开「分析报告」，等待分析结果准备完成。应用会复用已完成的计算，只补算缺失的分析。
2. 可以在「报告要求」里填写想重点解释的指标，然后点击「生成分析报告」。
3. 核对草稿中的数据、引用和结论。有需要可用「修订报告」或「退回」处理，确认后点击「通过审核」→「导出 Markdown」。报告的高风险处理方式可在「系统设置」→「报告安全」配置。

## 常见问题

| 现象 | 排查方式 |
| --- | --- |
| 获取模型失败 | 核对 API 地址和密钥。服务方未提供模型列表时，可查其文档并手动输入模型 ID |
| 获取到模型，但测试连接失败 | 获取模型使用列表接口，测试连接会实际请求模型生成。根据状态提示核对模型权限、接口协议和服务状态 |
| 报告提示上下文或输出长度超限 | 完整报告需要容纳分析摘要、模型思考和正文。选择上下文更大的模型，或在本机模型服务中调大上下文与输出额度 |
| 返回 HTTP 502 | 网关未收到有效的上游模型响应。检查所选服务的状态，恢复后重新测试连接 |
| 导入 CSV 失败 | 核对表头、列值范围和缺失值格式。执行分类模型评估时，数据需要同时包含阳性与阴性样本 |

## 项目结构与测试

| 路径 | 内容 |
| --- | --- |
| [`main.py`](main.py) | 应用启动入口 |
| [`ui/`](ui/) | PySide6 界面、图表、工作台与系统设置 |
| [`algorithm/`](algorithm/) | 数据加载、统计、模型训练和评估 |
| [`ai/`](ai/) | 模型服务配置、报告编排、知识检索与审查 |
| [`tests/`](tests/) 和 [`algorithm/tests/`](algorithm/tests/) | 应用流程与算法回归测试 |

安装项目依赖后，可用下面的命令运行测试。统计方法和指标定义见算法文档。[算法文档](algorithm/README.md)

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -q
```
