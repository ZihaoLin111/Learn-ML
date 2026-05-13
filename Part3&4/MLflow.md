## 什么是MLflow？
> MLflow 是一个开源平台，专门为协助机器学习从业者和团队处理机器学习过程的复杂性而构建。MLflow 专注于机器学习项目的完整生命周期，确保每个阶段都可管理、可追溯且可重现。

## Autolog
``` python
def mlflow_autolog():
    # 自动将常用的训练参数/指标等记录到MLflow中
    mlflow.pytorch.autolog()
```