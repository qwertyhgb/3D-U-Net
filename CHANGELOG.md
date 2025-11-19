# 更新日志

## [1.1.0] - 2025-11-19

### 🎉 重大改进

#### 训练系统优化
- ✅ 添加训练进度条显示（tqdm）
- ✅ 优化 DataLoader 配置（pin_memory, persistent_workers）
- ✅ 改进模型保存格式（包含完整训练状态）
- ✅ 增强训练日志输出（实时显示指标）
- ✅ 优化训练曲线可视化（3 子图布局）
- ✅ 添加训练时间统计

#### 数据处理优化
- ✅ 添加数据集随机打乱功能（可复现）
- ✅ 改进 K 折划分逻辑
- ✅ 添加数据验证和错误提示
- ✅ 优化 Windows 系统 num_workers 配置

#### 新增功能
- ✅ 创建 `run_training.py` 训练启动脚本
- ✅ 创建 `run_evaluation.py` 评估启动脚本
- ✅ 添加快速测试模式（--quick-test）
- ✅ 支持命令行参数配置
- ✅ 添加训练确认提示

#### 文档完善
- ✅ 创建 `PROJECT_STRUCTURE.md` 项目结构文档
- ✅ 更新 `README.md` 添加徽章和详细说明
- ✅ 创建 `.gitignore` 文件
- ✅ 更新 `requirements.txt` 添加更多依赖

#### 依赖更新
- ✅ 添加 nibabel 支持
- ✅ 添加 seaborn 可视化
- ✅ 添加 pandas 数据处理
- ✅ 添加 scikit-learn 工具
- ✅ 使用版本范围而非固定版本

### 🐛 Bug 修复

#### 配置系统
- ✅ 修复 `evaluate.py` 配置导入错误
- ✅ 修复 `aggregate_results.py` 配置导入错误
- ✅ 统一配置访问方式

#### 数据增强
- ✅ 修复 TorchIO 轴顺序错误
- ✅ 修复多模态增强不同步问题
- ✅ 修复 `apply_augmentations_multi` 逻辑错误

#### 模型预测
- ✅ 修复 Sigmoid 重复应用问题
- ✅ 修复 TTA 预测输出范围错误

#### 其他修复
- ✅ 修复缺失的 `smoke_test.py` 脚本
- ✅ 添加错误处理和异常捕获
- ✅ 改进路径处理逻辑

### 📝 代码质量

- ✅ 所有文件通过语法检查
- ✅ 添加详细的函数文档
- ✅ 改进代码注释
- ✅ 统一代码风格

### 🔧 性能优化

- ✅ 启用 pin_memory 加速数据传输
- ✅ 使用 non_blocking 异步数据传输
- ✅ 优化 DataLoader workers 配置
- ✅ 改进内存使用效率

---

## [1.0.0] - 2025-11-18

### 初始版本
- ✅ 3D U-Net 模型实现
- ✅ K 折交叉验证训练
- ✅ 数据增强（传统 + TorchIO）
- ✅ 混合精度训练支持
- ✅ TensorBoard 集成
- ✅ 评估指标计算
- ✅ 预测和可视化
- ✅ 基础文档

---

## 未来计划

### v1.2.0
- [ ] 添加更多模型架构选项（Attention U-Net, nnU-Net）
- [ ] 支持 3D 数据增强库（Albumentations3D）
- [ ] 添加模型集成（Ensemble）功能
- [ ] 支持分布式训练
- [ ] 添加自动超参数调优

### v1.3.0
- [ ] Web 界面（Gradio/Streamlit）
- [ ] 模型压缩和量化
- [ ] ONNX 导出支持
- [ ] Docker 容器化
- [ ] CI/CD 集成

### v2.0.0
- [ ] 支持更多医学影像模态（CT, PET）
- [ ] 多任务学习（分割 + 分类）
- [ ] 迁移学习支持
- [ ] 联邦学习支持
- [ ] 云端训练支持

---

## 贡献者

- TraeAI Assistant - 初始开发和优化

## 反馈

如有问题或建议，请提交 Issue 或 Pull Request。
