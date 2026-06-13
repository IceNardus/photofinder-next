# PhotoFinder Next 架构重构指令

## 目标
将现有的 SuperPoint + LightGlue 特征匹配架构从"直接存全部描述子"优化为"Patch 聚合 + HNSW 粗筛 + LightGlue 精排"的两阶段检索架构。

---

## 一、数据模型设计

### 1.1 SQLite 表结构

```sql
-- Patch 特征缓存表 (完整 SuperPoint 特征，用于 LightGlue 精排)
CREATE TABLE patch_features (
    patch_id      TEXT PRIMARY KEY,      -- UUID
    image_id      TEXT NOT NULL,         -- 关联原图
    patch_index   INTEGER NOT NULL,      -- 1~4
    keypoints     BLOB NOT NULL,         -- [N, 2] f32 归一化坐标
    descriptors   BLOB NOT NULL,         -- [N, 256] f32
    num_keypoints INTEGER NOT NULL,      -- 实际关键点数量 (128~256)
    image_width   INTEGER NOT NULL,
    image_height  INTEGER NOT NULL,
    bbox_x        REAL NOT NULL,         -- Patch 在原图中的位置
    bbox_y        REAL NOT NULL,
    bbox_w        REAL NOT NULL,
    bbox_h        REAL NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES images(id)
);

-- Patch 聚合向量表 (用于 HNSW 粗筛)
CREATE TABLE patch_vectors (
    patch_id      TEXT PRIMARY KEY,
    image_id      TEXT NOT NULL,
    patch_index   INTEGER NOT NULL,
    vector        BLOB NOT NULL,         -- [256] f32 PCA/VLAD 聚合向量
    FOREIGN KEY (patch_id) REFERENCES patch_features(patch_id)
);

-- HNSW 索引元数据
CREATE TABLE hnsw_meta (
    id              INTEGER PRIMARY KEY,
    vector_dim      INTEGER NOT NULL,    -- 256
    patch_count     INTEGER NOT NULL,
    max_level       INTEGER NOT NULL,
    ef_construction INTEGER NOT NULL,
    M               INTEGER NOT NULL,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_patch_vectors_image ON patch_vectors(image_id);
CREATE INDEX idx_patch_features_image ON patch_features(image_id);
```

### 1.2 Rust 数据结构

```rust
// src/core/features/patch.rs

#[derive(Debug, Clone)]
pub struct PatchFeature {
    pub patch_id: String,
    pub image_id: String,
    pub patch_index: u8,
    pub keypoints: Vec<f32>,           // [N * 2] 归一化坐标
    pub descriptors: Vec<f32>,         // [N * 256]
    pub num_keypoints: usize,
    pub bbox: Bbox,
}

#[derive(Debug, Clone)]
pub struct PatchVector {
    pub patch_id: String,
    pub image_id: String,
    pub vector: Vec<f32>,              // [256] PCA/VLAD 聚合向量
}

#[derive(Debug, Clone)]
pub struct Bbox {
    pub x: f32, pub y: f32,
    pub w: f32, pub h: f32,
}
```

---

## 二、SuperPoint 特征聚合 (PCA/VLAD)

### 2.1 聚合算法

```rust
// src/ai/features/aggregation.rs

pub struct DescriptorAggregation {
    pca_matrix: Vec<f32>,  // [256, 64] PCA 投影矩阵 (可选)
    num_centroids: usize,  // VLAD k-means K 值
    centroids: Vec<f32>,   // [K, 256] VLAD 聚类中心
}

/// 将 N 个 256 维描述子聚合成 1 个聚合向量
///
/// # Arguments
/// * `descriptors` - [N, 256] 描述子矩阵
/// * `max_descriptors` - 最多使用的描述子数量 (128~256)
///
/// # Returns
/// * `[256]` 或 `[64]` 聚合向量
pub fn aggregate_descriptors(
    descriptors: &[f32],
    num_descriptors: usize,
    max_descriptors: usize,
) -> Vec<f32> {
    // 1. 选择置信度最高的 max_descriptors 个描述子
    // 2. 使用 VLAD: sum_i (desc_i - centroid[assignment_i])
    // 3. L2 归一化
}
```

### 2.2 关键点筛选

```rust
/// 从 SuperPoint 输出的 1024 个关键点中筛选最重要的 128~256 个
pub fn select_top_keypoints(
    keypoints: &[f32],      // [1024, 2]
    scores: &[f32],         // [1024] 置信度分数
    descriptors: &[f32],    // [1024, 256]
    max_keypoints: usize,  // 128~256
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // 按分数排序，取 top-k
}
```

---

## 三、离线建库流程

### 3.1 Patch 切分策略

```rust
// src/core/scanner.rs 修改

pub struct PatchConfig {
    pub patches_per_row: u8,    // 水平方向 Patch 数量 (1~2)
    pub patches_per_col: u8,    // 垂直方向 Patch 数量 (1~2)
    pub overlap_ratio: f32,     // Patch 重叠率 (0.0 ~ 0.3)
    pub max_keypoints_per_patch: usize,  // 每个 Patch 保留的关键点数
}

impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            patches_per_row: 2,
            patches_per_col: 2,
            overlap_ratio: 0.2,
            max_keypoints_per_patch: 256,
        }
    }
}

/// 将原图切分为 1~4 个 Patch
pub fn split_into_patches(
    image: &GrayImage,
    config: &PatchConfig,
) -> Vec<Patch> {
    // 使用滑动窗口切分，相邻 Patch 有 overlap
}
```

### 3.2 完整建库流程

```rust
// src/core/processing.rs 或新建 src/core/feature_extractor.rs

pub struct FeatureExtractor {
    superpoint_onnx: Session,      // SuperPoint ONNX Runtime Session
    aggregation: DescriptorAggregation,
    patch_config: PatchConfig,
}

impl FeatureExtractor {
    /// 处理单张图片，提取所有 Patch 特征
    pub fn process_image(&self, image_path: &str) -> Result<ImageFeatures> {
        // 1. 读取图片并转为灰度
        // 2. 切分为 Patches
        // 3. 对每个 Patch 运行 SuperPoint
        // 4. 筛选关键点 (按分数排序取 top-256)
        // 5. 对描述子进行 VLAD 聚合
        // 6. 返回 PatchFeature + PatchVector
    }
}

/// 单张图片的特征
pub struct ImageFeatures {
    pub image_id: String,
    pub patches: Vec<PatchFeature>,   // 完整特征 (存 SQLite)
    pub vectors: Vec<PatchVector>,    // 聚合向量 (存 HNSW)
}
```

---

## 四、在线检索流程

### 4.1 查询处理

```rust
// src/search/object_search.rs 修改

impl ObjectSearch {
    /// 使用截图查询相似物品
    pub async fn search_by_image(&self, query_image: &str) -> Result<Vec<SearchResult>> {
        // 1. 读取截图
        // 2. SuperPoint 提取关键点和描述子
        // 3. 筛选 top-256 关键点
        // 4. VLAD 聚合生成查询向量
        // 5. HNSW 粗筛 Top-K Patches
        // 6. Early Exit: 关键点数量预检
        // 7. LightGlue 精排 (从 SQLite 读取候选 Patch 特征)
        // 8. RANSAC 验证
        // 9. 按匹配数返回 Top-N 图片
    }
}
```

### 4.2 Early Exit 策略

```rust
// 在 LightGlue 精排前快速预检

pub fn early_exit_check(
    query_kpts: &[f32],
    query_desc_mean: &[f32],
    candidate_kpts: &[f32],
    candidate_desc_mean: &[f32],
) -> bool {
    // 1. 关键点数量差异 > 200 → 跳过
    if (query_kpts.len() as i32 - candidate_kpts.len() as i32).abs() > 200 {
        return true;
    }

    // 2. 描述子均值余弦相似度 < 0.3 → 跳过
    if cosine_sim(query_desc_mean, candidate_desc_mean) < 0.3 {
        return true;
    }

    // 3. 描述子方差差异 > 阈值 → 跳过
    // ...

    false  // 继续 LightGlue 精排
}
```

### 4.3 LightGlue 精排

```rust
// src/search/feature_matching.rs 新建

pub struct FeatureMatcher {
    lightglue_session: Session,   // LightGlue ONNX Runtime Session
}

impl FeatureMatcher {
    /// 匹配两组 SuperPoint 特征
    pub fn match_features(
        &self,
        kpts0: &[f32], desc0: &[f32],  // 查询截图
        kpts1: &[f32], desc1: &[f32],  // 候选 Patch
    ) -> MatchResult {
        // 1. 归一化关键点坐标到 [-1, 1]
        // 2. 组合成 [2, N, 2] 和 [2, N, 256] 格式
        // 3. 调用 LightGlue ONNX
        // 4. 解析 matches 和 mscores
        // 5. 过滤低置信度匹配
    }
}

#[derive(Debug)]
pub struct MatchResult {
    pub matches: Vec<Match>,
    pub scores: Vec<f32>,
}

#[derive(Debug)]
pub struct Match {
    pub kpt0_idx: usize,
    pub kpt1_idx: usize,
}
```

---

## 五、RANSAC 几何验证

```rust
// src/search/feature_matching.rs

/// 使用 RANSAC 过滤异常匹配
pub fn ransac_verify(
    kpts0: &[f32],   // [N, 2]
    kpts1: &[f32],   // [N, 2]
    matches: &[Match],
    inlier_threshold: f32,
    ransac_iterations: usize,
) -> Vec<Match> {
    // 1. 构建匹配点对
    // 2. 迭代 RANSAC:
    //    - 随机选择 4 个匹配点
    //    - 计算单应性矩阵 H
    //    - 计算内点数量
    // 3. 返回最优内点集合
}
```

---

## 六、HNSW 索引管理

```rust
// src/core/index/hnsw.rs 修改

pub struct HnswIndex {
    vectors: Vec<Vec<f32>>,          // 存储聚合向量
    patch_ids: Vec<String>,           // 对应的 patch_id
    // ... 现有 HNSW 结构
}

impl HnswIndex {
    /// 批量插入聚合向量
    pub fn insert_batch(&mut self, vectors: &[PatchVector]) {
        // 批量添加向量到 HNSW
    }

    /// 搜索 Top-K 最近邻
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<SearchResult> {
        // HNSW 搜索
    }
}
```

---

## 七、模型文件

| 模型 | 用途 | 位置 |
|------|------|------|
| `superpoint.onnx` | 关键点+描述子提取 | `resources/models/` |
| `lightglue.onnx` | 特征匹配 | `resources/models/` |
| `mobileclip_s2.onnx` | 语义嵌入 (可选) | `resources/models/` |

---

## 八、配置文件

```toml
# src-tauri/tauri.conf.json 或单独的 config.toml

[features]
superpoint_max_keypoints = 1024   # SuperPoint 最大关键点数
patch_max_keypoints = 256         # 每个 Patch 保留的关键点数
patches_per_row = 2               # 水平 Patch 数
patches_per_col = 2              # 垂直 Patch 数
patch_overlap = 0.2               # Patch 重叠率

[hnsw]
M = 16
ef_construction = 200
ef_search = 100

[matching]
lightglue_threshold = 0.1         # LightGlue 匹配阈值
ransac_threshold = 4.0           # RANSAC 内点阈值
ransac_iterations = 1000
min_inliers = 10                  # 最少内点数

[early_exit]
max_kpt_difference = 200          # 关键点数量差异阈值
min_desc_sim = 0.3                # 描述子均值最小相似度
```

---

## 九、修改文件清单

### 新建文件
- `src/core/features/mod.rs`
- `src/core/features/patch.rs`
- `src/core/features/aggregation.rs`
- `src/search/feature_matching.rs`

### 修改文件
- `src/core/database/schema.sql` - 添加新表
- `src/core/scanner.rs` - Patch 切分逻辑
- `src/core/processing.rs` - 特征提取流程
- `src/core/index/hnsw.rs` - HNSW 批量插入
- `src/search/object_search.rs` - 两阶段检索
- `src/ai/object/mod.rs` - 移除/简化 MobileNet 依赖

### 配置文件
- `src-tauri/tauri.conf.json` - 添加 feature 配置

---

## 十、实现顺序

1. **阶段1: 数据模型**
   - 定义 Rust 数据结构 `PatchFeature`, `PatchVector`
   - 更新 SQLite schema

2. **阶段2: SuperPoint 封装**
   - 封装 SuperPoint ONNX 调用
   - 实现关键点筛选 (按分数排序取 top-256)

3. **阶段3: 聚合算法**
   - 实现 VLAD 聚合
   - 实现描述子均值/方差计算

4. **阶段4: Patch 切分**
   - 实现图片 Patch 切分
   - 集成到 Scanner

5. **阶段5: LightGlue 封装**
   - 封装 LightGlue ONNX 调用
   - 实现 keypoint 归一化

6. **阶段6: 精排流程**
   - 实现 Early Exit
   - 实现 LightGlue 精排
   - 实现 RANSAC 验证

7. **阶段7: 集成测试**
   - 端到端测试
   - 性能调优
