import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

interface Statistics {
  image_count: number;
  face_count: number;
  patch_count: number;
  object_count: number;
  pending_task_count: number;
  thumbnail_count: number;
  thumbnail_size_bytes: number;
}

interface ScanStatus {
  is_scanning: boolean;
  total_images: number;
  processed_images: number;
  pending_tasks: number;
  current_file: string;
}

interface ProcessingStatus {
  current_image: string;
  current_faces: number;
  current_patches: number;
  current_objects: number;
  log_message: string;
  last_completion_message: string;
}

interface SearchResult {
  image_id: number;
  face_id?: number;
  thumbnail_path: string;
  similarity: number;
  bbox?: number[];
}

let currentPage = "person";
let isScanning = false;
let personImagePath: string | null = null;
let lastLogMessage = "";
let lastCompletionMessage = "";

// Person search export state
let selectedResults: Set<number> = new Set();
let exportDir: string | null = null;
const RESULTS_PER_PAGE = 20;
let currentPageNum = 1;
let totalResults = 0;

// Track state for potential future use
void currentPage;
void isScanning;

// Navigation
document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const page = (btn as HTMLElement).dataset.page;
    if (page) switchPage(page);
  });
});

// Initialize page visibility on load
switchPage("scan");

function switchPage(page: string) {
  currentPage = page;
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.classList.toggle("active", (btn as HTMLElement).dataset.page === page);
  });
  document.querySelectorAll(".page").forEach((p) => {
    p.classList.toggle("active", p.id === `page-${page}`);
  });
}

// Statistics - uses updateScanProgress to also update progress bar
async function loadStatistics() {
  await updateScanProgress();
}

function formatNumber(n: number): string {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
  if (n >= 1000) return (n / 1000).toFixed(1) + "K";
  return n.toString();
}

// Scan page elements - scan page HTML was removed, so use optional chaining
// These are kept for reference but scan page functionality is disabled
const btnAddFolder = document.getElementById("btn-add-folder") as HTMLButtonElement | null;
const btnStartScan = document.getElementById("btn-start-scan") as HTMLButtonElement | null;
const btnStopScan = document.getElementById("btn-stop-scan") as HTMLButtonElement | null;
const progressFill = document.getElementById("progress-fill");
const progressCount = document.getElementById("progress-count");
const progressImages = document.getElementById("progress-images");
const progressTasks = document.getElementById("progress-tasks");
const progressTitle = document.querySelector(".progress-title");
const scanLog = document.getElementById("scan-log");
const btnClearLog = document.getElementById("btn-clear-log");
const btnClearDatabase = document.getElementById("btn-clear-database") as HTMLButtonElement;

// Folder management
const folderList = document.getElementById("folder-list") as HTMLDivElement;
let folders: string[] = [];

let scanStatusInterval: number | null = null;
let logEntries: { time: Date; type: string; message: string }[] = [];

// Scan button event listeners
btnAddFolder?.addEventListener("click", async () => {
  try {
    const selected = await open({
      directory: true,
      multiple: false,
    });
    if (selected && typeof selected === "string") {
      if (!folders.includes(selected)) {
        folders.push(selected);
        renderFolderList();
        addLog("info", `已添加文件夹: ${selected}`);
        if (btnStartScan) btnStartScan.disabled = false;
      }
    }
  } catch (e) {
    console.error("添加文件夹失败:", e);
    addLog("error", `添加文件夹失败: ${e}`);
  }
});

btnClearDatabase?.addEventListener("click", async () => {
  if (!confirm("确定要清空所有数据吗？这将删除所有图片索引、人脸数据、缩略图和搜索索引。此操作不可撤销。")) {
    return;
  }
  try {
    if (btnClearDatabase) btnClearDatabase.disabled = true;
    addLog("info", "正在清空数据库...");
    const result = await invoke<any>("clear_database");
    if (result.success) {
      addLog("success", "数据库已清空");
      folders = [];
      renderFolderList();
      if (btnStartScan) btnStartScan.disabled = true;
      await loadStatistics();
    } else {
      addLog("error", `清空失败: ${result.errors.join(", ")}`);
    }
  } catch (e) {
    console.error("清空数据库失败:", e);
    addLog("error", `清空数据库失败: ${e}`);
  } finally {
    if (btnClearDatabase) btnClearDatabase.disabled = false;
  }
});

btnStartScan?.addEventListener("click", async () => {
  if (folders.length === 0) {
    addLog("warning", "请先添加要扫描的文件夹");
    return;
  }
  try {
    isScanning = true;
    if (btnStartScan) btnStartScan.disabled = true;
    if (btnStopScan) btnStopScan.disabled = false;
    if (btnAddFolder) btnAddFolder.disabled = true;
    addLog("info", "开始扫描...");

    // Reset log tracking
    lastLogMessage = "";
    lastCompletionMessage = "";

    // Show progress container
    const progressContainer = document.getElementById("progress-container");
    if (progressContainer) progressContainer.style.display = "block";

    for (const folder of folders) {
      await invoke("scan_folder", { folderPath: folder });
    }

    // Keep polling for processing progress after scan completes
    scanStatusInterval = window.setInterval(updateScanProgress, 1000);
  } catch (e) {
    console.error("扫描失败:", e);
    addLog("error", `扫描失败: ${e}`);
    isScanning = false;
    if (btnStartScan) btnStartScan.disabled = false;
    if (btnStopScan) btnStopScan.disabled = true;
    if (btnAddFolder) btnAddFolder.disabled = false;
  }
});

btnStopScan?.addEventListener("click", async () => {
  try {
    await invoke("stop_scan");
    addLog("warning", "已停止扫描");
    isScanning = false;
    if (btnStartScan) btnStartScan.disabled = false;
    if (btnStopScan) btnStopScan.disabled = true;
    if (btnAddFolder) btnAddFolder.disabled = false;
    if (scanStatusInterval) {
      clearInterval(scanStatusInterval);
      scanStatusInterval = null;
    }
  } catch (e) {
    console.error("停止扫描失败:", e);
  }
});

function renderFolderList() {
  if (!folderList) return;

  if (folders.length === 0) {
    folderList.innerHTML = `
      <div class="empty-state">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
          <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/>
        </svg>
        <p>尚未添加任何文件夹</p>
        <p class="hint">点击"添加文件夹"开始扫描照片</p>
      </div>
    `;
    return;
  }

  folderList.innerHTML = folders.map(f => `
    <div class="folder-item">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/>
      </svg>
      <span class="path" title="${f}">${f}</span>
      <button class="btn-remove" data-folder="${f}">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M18 6L6 18M6 6l12 12"/>
        </svg>
      </button>
    </div>
  `).join('');

  // Add remove listeners
  folderList.querySelectorAll('.btn-remove').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const folder = (e.currentTarget as HTMLElement).dataset.folder;
      if (folder) {
        folders = folders.filter(f => f !== folder);
        renderFolderList();
        if (btnStartScan) btnStartScan.disabled = folders.length === 0;
        addLog("info", `已移除文件夹: ${folder}`);
      }
    });
  });
}

function addLog(type: "info" | "success" | "warning" | "error", message: string) {
  const time = new Date();
  logEntries.push({ time, type, message });

  // Keep only last 100 entries
  if (logEntries.length > 100) {
    logEntries = logEntries.slice(-100);
  }

  const entry = document.createElement("div");
  entry.className = `log-entry ${type}`;
  const timeStr = time.toLocaleTimeString("zh-CN", { hour12: false });
  entry.innerHTML = `<span class="log-time">${timeStr}</span>${message}`;
  scanLog?.appendChild(entry);

  // Auto scroll to bottom
  scanLog?.scrollTo(0, scanLog.scrollHeight);
}

btnClearLog?.addEventListener("click", () => {
  logEntries = [];
  if (scanLog) scanLog.innerHTML = "";
});

async function updateScanProgress() {
  try {
    const status = await invoke<ScanStatus>("get_scan_status");
    const procStatus = await invoke<ProcessingStatus>("get_processing_status");
    const stats = await invoke<Statistics>("get_statistics");

    // Update sidebar stats (always work, even without scan page)
    document.getElementById("stat-images")!.textContent = formatNumber(stats.image_count);
    document.getElementById("stat-faces")!.textContent = formatNumber(stats.face_count);
    document.getElementById("stat-objects")!.textContent = formatNumber(stats.object_count);

    // Add new log entries from scan status (files being scanned)
    if (status.current_file && status.is_scanning) {
      const scanLogMsg = `扫描: ${status.current_file}`;
      if (scanLogMsg !== lastLogMessage) {
        addLog("info", scanLogMsg);
        lastLogMessage = scanLogMsg;
      }
    }

    // Add new log entries from processing (completed images with face/object counts)
    if (procStatus.last_completion_message && procStatus.last_completion_message !== lastCompletionMessage) {
      addLog("success", procStatus.last_completion_message);
      lastCompletionMessage = procStatus.last_completion_message;
      lastLogMessage = procStatus.last_completion_message; // Avoid duplicate
    }

    // Only update scan progress UI if scan page elements exist
    if (status.is_scanning && progressTitle) {
      progressTitle.textContent = "扫描中...";
      const total = status.total_images;
      const processed = status.processed_images;
      const percent = total > 0 ? (processed / total) * 100 : 0;
      if (progressFill) progressFill.style.width = `${percent}%`;
      if (progressCount) progressCount.textContent = `${processed} / ${total}`;
      if (progressImages) progressImages.textContent = `已添加: ${processed} 张图片`;
      const faceInfo = procStatus.current_faces > 0 ? `人脸${procStatus.current_faces}` : "";
      const objInfo = procStatus.current_objects > 0 ? `物品${procStatus.current_objects}` : "";
      const detailInfo = [faceInfo, objInfo].filter(Boolean).join(", ");
      if (progressTasks) progressTasks.textContent = detailInfo ? `检测: ${detailInfo}` : `待处理: ${status.pending_tasks} 个任务`;
      if (status.current_file && progressTasks) {
        const fileName = status.current_file.split("/").pop() || status.current_file;
        progressTasks.textContent += ` | ${fileName}`;
      }
    } else if (!status.is_scanning && progressTitle) {
      const total = stats.image_count;
      const pending = status.pending_tasks;
      const processed = total - pending;
      const percent = total > 0 ? (processed / total) * 100 : 100;
      if (progressFill) progressFill.style.width = `${percent}%`;
      if (progressCount) progressCount.textContent = `${processed} / ${total}`;
      if (progressImages) progressImages.textContent = `已处理: ${processed} 张图片`;
      if (progressTasks) progressTasks.textContent = pending > 0 ? `待处理: ${pending} 张` : "处理完成";
      if (pending === 0 && scanStatusInterval) {
        clearInterval(scanStatusInterval);
        scanStatusInterval = null;
        isScanning = false;
        if (btnStartScan) btnStartScan.disabled = false;
        if (btnStopScan) btnStopScan.disabled = true;
        if (btnAddFolder) btnAddFolder.disabled = false;
        addLog("success", "所有图片处理完成!");
      }
    }
  } catch (e) {
    console.error("获取扫描状态失败:", e);
  }
}

// Person Search
const personUpload = document.getElementById("person-upload")!;
const personFileInput = document.getElementById("person-file-input") as HTMLInputElement;
const personPreview = document.getElementById("person-preview")!;
const personPreviewImg = document.getElementById("person-preview-img") as HTMLImageElement;
const btnPersonSearch = document.getElementById("btn-person-search") as HTMLButtonElement;
const btnPersonClear = document.getElementById("btn-person-clear")!;
const personResults = document.getElementById("person-results")!;
const personPagination = document.getElementById("person-pagination")!;

// Event delegation for result card clicks
personResults.addEventListener("click", (e) => {
  const target = e.target as Element;
  const card = target.closest('.result-card');

  if (card) {
    const indexStr = card.getAttribute('data-index');
    if (indexStr !== null) {
      toggleResultSelection(parseInt(indexStr, 10));
    }
  }
});

// Event delegation for pagination clicks
personPagination.addEventListener("click", (e) => {
  const btn = (e.target as HTMLElement).closest('.page-btn');
  if (btn && !(btn as HTMLButtonElement).disabled) {
    const page = parseInt((btn as HTMLElement).dataset.page!, 10);
    goToPage(page);
  }
});

personUpload.addEventListener("click", () => personFileInput.click());
personUpload.addEventListener("dragover", (e) => {
  e.preventDefault();
  personUpload.style.borderColor = "var(--accent)";
});
personUpload.addEventListener("dragleave", () => {
  personUpload.style.borderColor = "";
});
personUpload.addEventListener("drop", (e) => {
  e.preventDefault();
  personUpload.style.borderColor = "";
  const file = e.dataTransfer?.files[0];
  if (file) handlePersonFile(file);
});

personFileInput.addEventListener("change", () => {
  const file = personFileInput.files?.[0];
  if (file) handlePersonFile(file);
});

async function handlePersonFile(file: File) {
  // Read file as base64 data URL for preview
  personImagePath = file.name;
  const reader = new FileReader();
  reader.onload = () => {
    personPreviewImg.src = reader.result as string;
    personPreview.style.display = "block";
    personUpload.style.display = "none";
    btnPersonSearch.disabled = false;
  };
  reader.readAsDataURL(file);
}

btnPersonSearch.addEventListener("click", async () => {
  if (!personImagePath) return;
  btnPersonSearch.disabled = true;
  personResults.innerHTML = '<p style="color: var(--text-secondary)">搜索中...</p>';
  try {
    // Convert base64 to bytes
    const dataUrl = personPreviewImg.src;
    const base64Data = dataUrl.split(',')[1];
    const mimeType = dataUrl.split(';')[0].split(':')[1]; // e.g. "image/jpeg"
    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    await invoke("write_query_image", { data: Array.from(bytes), mimeType });
    console.log("[SEARCH] Wrote query image ({}), starting search...", mimeType);

    // Determine extension from mime type
    const ext = mimeType === "image/png" ? "png" : mimeType === "image/webp" ? "webp" : "jpg";
    const queryPath = `PhotoFinderNext/query_image.${ext}`;

    const results = await invoke<SearchResult[]>("search_person", {
      queryImage: queryPath,
      topK: 50,
    });
    console.log("[SEARCH] Results:", results);
    await renderSearchResults(results, personResults);
  } catch (e) {
    console.error("搜索失败:", e);
    personResults.innerHTML = `<p style="color: var(--danger)">搜索失败: ${e}</p>`;
  }
  btnPersonSearch.disabled = false;
});

btnPersonClear.addEventListener("click", () => {
  personImagePath = null;
  personPreview.style.display = "none";
  personUpload.style.display = "flex";
  btnPersonSearch.disabled = true;
  personResults.innerHTML = "";
  selectedResults.clear();
  currentSearchResults = [];
  document.getElementById("person-export-controls")!.style.display = 'none';
  document.getElementById("person-pagination")!.innerHTML = '';
});

// Export controls
const btnChooseExportDir = document.getElementById("btn-choose-export-dir") as HTMLButtonElement;
const btnPersonExport = document.getElementById("btn-person-export") as HTMLButtonElement;
const personSelectAll = document.getElementById("person-select-all") as HTMLInputElement;

btnChooseExportDir?.addEventListener("click", async () => {
  try {
    const selected = await open({ directory: true });
    if (selected) {
      exportDir = selected as string;
      document.getElementById("export-dir-path")!.textContent = exportDir;
      updateExportControls();
    }
  } catch (e) {
    console.error("选择导出目录失败:", e);
  }
});

btnPersonExport?.addEventListener("click", async () => {
  if (selectedResults.size === 0 || !exportDir) return;
  const selectedPaths: string[] = [];
  for (const idx of selectedResults) {
    if (currentSearchResults[idx]?.thumbnail_path) {
      selectedPaths.push(currentSearchResults[idx].thumbnail_path);
    }
  }
  if (selectedPaths.length > 0) {
    try {
      const results = await invoke<string[]>("copy_files", { sourcePaths: selectedPaths, destDir: exportDir });
      alert(`成功复制 ${results.length} 个文件到: ${exportDir}`);
      console.log("[EXPORT] Copied files:", results);
    } catch (e) {
      alert(`复制失败: ${e}`);
      console.error("[EXPORT] Copy failed:", e);
    }
  }
});

personSelectAll?.addEventListener("change", () => {
  if (personSelectAll.checked) {
    // Select all visible results
    const totalPages = Math.ceil(totalResults / RESULTS_PER_PAGE);
    for (let p = 1; p <= totalPages; p++) {
      const start = (p - 1) * RESULTS_PER_PAGE;
      const end = Math.min(start + RESULTS_PER_PAGE, totalResults);
      for (let i = start; i < end; i++) {
        selectedResults.add(i);
      }
    }
  } else {
    selectedResults.clear();
  }
  updateExportControls();
  renderResultsPage();
});
// Store current search results for export
let currentSearchResults: SearchResult[] = [];

async function renderSearchResults(results: SearchResult[], container: HTMLElement) {
  currentSearchResults = results;
  totalResults = results.length;
  currentPageNum = 1;
  selectedResults.clear();
  updateExportControls();

  if (results.length === 0) {
    container.innerHTML = '<p style="color: var(--text-secondary)">未找到结果</p>';
    document.getElementById("person-pagination")!.innerHTML = '';
    return;
  }

  // Show export controls
  document.getElementById("person-export-controls")!.style.display = 'flex';
  container.innerHTML = `<p style="color: var(--text-secondary)">加载图片中... (${results.length} 结果)</p>`;

  console.log("[SEARCH] renderSearchResults called with", results.length, "results");

  try {
    await renderResultsPage();
  } catch (e) {
    console.error("[SEARCH] render error:", e);
    container.innerHTML = `<p style="color: var(--danger)">显示失败: ${e}</p>`;
  }
}

async function renderResultsPage() {
  const personResults = document.getElementById("person-results")!;
  const pagination = document.getElementById("person-pagination")!;

  const startIdx = (currentPageNum - 1) * RESULTS_PER_PAGE;
  const endIdx = Math.min(startIdx + RESULTS_PER_PAGE, totalResults);
  const pageResults = currentSearchResults.slice(startIdx, endIdx);

  const htmlParts = [];
  for (let i = 0; i < pageResults.length; i++) {
    const r = pageResults[i];
    const globalIdx = startIdx + i;
    const filename = r.thumbnail_path ? r.thumbnail_path.split('/').pop() || 'unknown' : 'unknown';
    const displayName = filename.length > 20 ? filename.slice(0, 17) + '...' : filename;
    const isSelected = selectedResults.has(globalIdx);

    // Load thumbnail via Tauri command
    let imgTag = '';
    console.log("[RENDER] Loading thumbnail for:", r.thumbnail_path);
    if (r.thumbnail_path) {
      try {
        const base64 = await Promise.race([
          invoke<string>("get_image_thumbnail", { imagePath: r.thumbnail_path }),
          new Promise((_, reject) => setTimeout(() => reject(new Error("Timeout")), 5000))
        ]);
        imgTag = `<img src="${base64}" alt="${displayName}" style="width:100%;aspect-ratio:1;object-fit:cover;" />`;
      } catch (e) {
        console.error("[RENDER] Thumbnail error:", e, "path:", r.thumbnail_path);
        imgTag = `<div style="background:linear-gradient(135deg,#667eea,#764ba2);display:flex;align-items:center;justify-content:center;color:white;aspect-ratio:1;font-size:12px;">ERR</div>`;
      }
    } else {
      console.log("[RENDER] No thumbnail_path for result:", globalIdx);
      imgTag = `<div style="background:linear-gradient(135deg,#667eea,#764ba2);display:flex;align-items:center;justify-content:center;color:white;aspect-ratio:1;font-size:12px;">NO PATH</div>`;
    }

    htmlParts.push(`
      <div class="result-card" data-index="${globalIdx}" style="cursor:pointer;">
        <div class="checkbox-overlay ${isSelected ? 'checked' : ''}" data-result-index="${globalIdx}">
          ${isSelected ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"></polyline></svg>' : ''}
        </div>
        ${imgTag}
        <div class="result-info">
          <div class="result-filename" title="${r.thumbnail_path || ''}">${displayName}</div>
          <div class="result-similarity" style="font-size:11px;color:${r.similarity >= 0.7 ? '#4ade80' : r.similarity >= 0.5 ? '#facc15' : '#f87171'};">
            相似度: ${(r.similarity * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    `);
  }
  console.log("[RENDER] All thumbnails loaded, setting innerHTML");
  personResults.innerHTML = htmlParts.join('');
  console.log("[RENDER] Done");

  // Render pagination
  const totalPages = Math.ceil(totalResults / RESULTS_PER_PAGE);
  let paginationHtml = '';
  if (totalPages > 1) {
    paginationHtml = `
      <button class="page-btn" data-page="${currentPageNum - 1}" ${currentPageNum === 1 ? 'disabled' : ''}>上一页</button>
      <span class="page-info">第 ${currentPageNum} / ${totalPages} 页</span>
      <button class="page-btn" data-page="${currentPageNum + 1}" ${currentPageNum === totalPages ? 'disabled' : ''}>下一页</button>
    `;
  }
  pagination.innerHTML = paginationHtml;
}

function toggleResultSelection(index: number) {
  if (selectedResults.has(index)) {
    selectedResults.delete(index);
  } else {
    selectedResults.add(index);
  }
  updateExportControls();
  renderResultsPage();
}

function updateExportControls() {
  const count = selectedResults.size;
  document.getElementById("person-selected-count")!.textContent = `已选择: ${count}`;
  document.getElementById("export-count")!.textContent = count.toString();
  (document.getElementById("btn-person-export") as HTMLButtonElement).disabled = count === 0 || !exportDir;
}

function goToPage(page: number) {
  const totalPages = Math.ceil(totalResults / RESULTS_PER_PAGE);
  if (page >= 1 && page <= totalPages) {
    currentPageNum = page;
    renderResultsPage();
  }
}

// Make functions globally accessible
(window as any).toggleResultSelection = toggleResultSelection;
(window as any).goToPage = goToPage;

// Dev Test buttons
const btnTestPipeline = document.getElementById("btn-test-pipeline");
const btnTestSimilarity = document.getElementById("btn-test-similarity");
const devResult = document.getElementById("dev-result");

btnTestPipeline?.addEventListener("click", async () => {
  const path = "/Users/mac/Downloads/asian-beautiful-woman-with-brown-long-hair-portrait-white-tshirt-jean-jacket-costume-liftstyle-concept.jpg";
  if (devResult) { devResult.textContent = "测试中..."; }
  try {
    const result = await invoke<any>("test_face_pipeline", { imagePath: path });
    if (devResult) { devResult.textContent = JSON.stringify(result, null, 2); }
  } catch (e) {
    if (devResult) { devResult.textContent = "Error: " + e; }
  }
});

btnTestSimilarity?.addEventListener("click", async () => {
  const path1 = "/Users/mac/Downloads/jJSdALjbCewl05W.thumb.1000_0.jpg";
  const path2 = "/Users/mac/Downloads/73S2Y9lgheaynz0.thumb.1000_0.jpg";
  if (devResult) { devResult.textContent = "测试中..."; }
  try {
    const result = await invoke<any>("test_face_similarity", { image1Path: path1, image2Path: path2 });
    if (devResult) { devResult.textContent = JSON.stringify(result, null, 2); }
  } catch (e) {
    if (devResult) { devResult.textContent = "Error: " + e; }
  }
});

const btnRebuildFaceIndex = document.getElementById("btn-rebuild-face-index") as HTMLButtonElement;
btnRebuildFaceIndex?.addEventListener("click", async () => {
  if (devResult) { devResult.textContent = "重建人脸索引中..."; }
  try {
    const result = await invoke<any>("rebuild_face_index");
    if (devResult) { devResult.textContent = JSON.stringify(result); }
  } catch (e) {
    if (devResult) { devResult.textContent = "Error: " + e; }
  }
});

const btnDebugFaces = document.getElementById("btn-debug-faces") as HTMLButtonElement;
btnDebugFaces?.addEventListener("click", async () => {
  if (devResult) { devResult.textContent = "Debugging..."; }
  try {
    const result = await invoke<any>("debug_faces");
    if (devResult) { devResult.textContent = JSON.stringify(result); }
  } catch (e) {
    if (devResult) { devResult.textContent = "Error: " + e; }
  }
});


const objectUpload = document.getElementById("object-upload")!;
const objectFileInput = document.getElementById("object-file-input") as HTMLInputElement;
const objectPreview = document.getElementById("object-preview")!;
const objectPreviewImg = document.getElementById("object-preview-img") as HTMLImageElement;
const objectRoisInfo = document.getElementById("object-rois-info")!;
const btnObjectSearch = document.getElementById("btn-object-search") as HTMLButtonElement;
const btnObjectClear = document.getElementById("btn-object-clear")!;
const objectResults = document.getElementById("object-results")!;

// Crop modal elements
const cropModal = document.getElementById("crop-modal")!;
const cropPreviewImg = document.getElementById("crop-preview-img") as HTMLImageElement;
const cropBox = document.getElementById("crop-box")!;
const btnCropSearch = document.getElementById("btn-crop-search") as HTMLButtonElement;
const btnCropClear = document.getElementById("btn-crop-clear") as HTMLButtonElement;
const btnCropBack = document.getElementById("btn-crop-back") as HTMLButtonElement;
const btnCloseCropModal = document.getElementById("btn-close-crop-modal") as HTMLButtonElement;

let objectImagePath: string | null = null;
let objectFileData: Uint8Array | null = null;  // Store original file data for cropping
let cropStartX = 0;
let cropStartY = 0;
let isCropping = false;
let currentObjectResults: any[] = [];
let objectPageNum = 1;
let objectTotalResults = 0;

// Object search export state
let selectedObjectResults: Set<number> = new Set();
let objectExportDir: string | null = null;

objectUpload.addEventListener("click", () => objectFileInput.click());
objectUpload.addEventListener("dragover", (e) => {
  e.preventDefault();
  objectUpload.style.borderColor = "var(--accent)";
});
objectUpload.addEventListener("dragleave", () => {
  objectUpload.style.borderColor = "";
});
objectUpload.addEventListener("drop", (e) => {
  e.preventDefault();
  objectUpload.style.borderColor = "";
  const file = e.dataTransfer?.files[0];
  if (file && file.type.startsWith("image/")) {
    handleObjectFile(file);
  }
});

objectFileInput.addEventListener("change", () => {
  const file = objectFileInput.files?.[0];
  if (file) handleObjectFile(file);
});

async function handleObjectFile(file: File) {
  try {
    const arrayBuffer = await file.arrayBuffer();
    objectFileData = new Uint8Array(arrayBuffer);

    // Create URL for preview
    const url = URL.createObjectURL(file);

    // Open crop modal with the image
    cropPreviewImg.src = url;
    cropModal.style.display = "flex";
    cropBox.style.display = "none";
    btnCropSearch.disabled = true;

    // Reset crop selection
    cropStartX = 0;
    cropStartY = 0;
  } catch (e) {
    objectRoisInfo.textContent = "图片加载失败: " + e;
  }
}

// Crop interaction handlers
let cropLeft = 0, cropTop = 0, cropWidth = 0, cropHeight = 0;
let isDragging = false, isResizing = false;
let resizeHandle = "";
let dragOffsetX = 0, dragOffsetY = 0;

function updateCropBox() {
  const rect = cropPreviewImg.getBoundingClientRect();
  const scaleX = cropPreviewImg.naturalWidth / rect.width;
  const scaleY = cropPreviewImg.naturalHeight / rect.height;

  cropBox.style.left = (cropLeft / scaleX) + "px";
  cropBox.style.top = (cropTop / scaleY) + "px";
  cropBox.style.width = (cropWidth / scaleX) + "px";
  cropBox.style.height = (cropHeight / scaleY) + "px";
  cropBox.style.display = "block";

  // Update dimensions display
  let dimLabel = cropBox.querySelector(".crop-dimensions") as HTMLElement;
  if (!dimLabel) {
    dimLabel = document.createElement("div");
    dimLabel.className = "crop-dimensions";
    cropBox.appendChild(dimLabel);
  }
  dimLabel.textContent = `${Math.round(cropWidth)} × ${Math.round(cropHeight)}`;

  btnCropSearch.disabled = cropWidth < 10 || cropHeight < 10;
}

function createHandles() {
  cropBox.innerHTML = "";
  const handles = ["nw", "ne", "sw", "se", "n", "s", "e", "w"];
  handles.forEach(h => {
    const handle = document.createElement("div");
    handle.className = `handle ${h}`;
    handle.dataset.handle = h;
    cropBox.appendChild(handle);
  });
}

function getHandleAtPoint(x: number, y: number): string | null {
  const handles = cropBox.querySelectorAll(".handle");
  for (const handle of handles) {
    const rect = (handle as HTMLElement).getBoundingClientRect();
    if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
      return (handle as HTMLElement).dataset.handle || null;
    }
  }
  return null;
}

cropPreviewImg.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  if (!cropPreviewImg.naturalWidth || !cropPreviewImg.naturalHeight) {
    console.log("[CROP] Image not loaded yet");
    return;
  }

  const rect = cropPreviewImg.getBoundingClientRect();
  const scaleX = cropPreviewImg.naturalWidth / rect.width;
  const scaleY = cropPreviewImg.naturalHeight / rect.height;
  const mouseX = (e.clientX - rect.left) * scaleX;
  const mouseY = (e.clientY - rect.top) * scaleY;

  console.log("[CROP] mousedown: rect=", rect, "natural=", cropPreviewImg.naturalWidth, cropPreviewImg.naturalHeight);
  console.log("[CROP] mousedown: mouseX=", mouseX, "mouseY=", mouseY);

  // Check if clicking on a resize handle
  const handle = getHandleAtPoint(e.clientX, e.clientY);
  if (handle) {
    isResizing = true;
    resizeHandle = handle;
    e.preventDefault();
    return;
  }

  // Check if clicking inside existing crop box
  const boxLeft = parseFloat(cropBox.style.left || "0") * scaleX;
  const boxTop = parseFloat(cropBox.style.top || "0") * scaleY;
  const boxWidth = parseFloat(cropBox.style.width || "0") * scaleX;
  const boxHeight = parseFloat(cropBox.style.height || "0") * scaleY;

  if (boxWidth > 0 && boxHeight > 0 &&
      mouseX >= boxLeft && mouseX <= boxLeft + boxWidth &&
      mouseY >= boxTop && mouseY <= boxTop + boxHeight) {
    isDragging = true;
    dragOffsetX = mouseX - boxLeft;
    dragOffsetY = mouseY - boxTop;
    e.preventDefault();
    return;
  }

  // Start new crop
  cropStartX = mouseX;
  cropStartY = mouseY;
  isCropping = true;
  createHandles();
});

cropPreviewImg.addEventListener("mousemove", (e) => {
  if (!cropPreviewImg.naturalWidth || !cropPreviewImg.naturalHeight) return;

  const rect = cropPreviewImg.getBoundingClientRect();
  const scaleX = cropPreviewImg.naturalWidth / rect.width;
  const scaleY = cropPreviewImg.naturalHeight / rect.height;
  const mouseX = (e.clientX - rect.left) * scaleX;
  const mouseY = (e.clientY - rect.top) * scaleY;

  // Clamp mouse coordinates to image bounds
  const clampedMouseX = Math.max(0, Math.min(mouseX, cropPreviewImg.naturalWidth));
  const clampedMouseY = Math.max(0, Math.min(mouseY, cropPreviewImg.naturalHeight));

  if (isResizing) {
    // Handle resize
    let newLeft = cropLeft, newTop = cropTop, newWidth = cropWidth, newHeight = cropHeight;
    const minSize = 20;

    if (resizeHandle.includes("e")) {
      newWidth = Math.max(minSize, Math.min(clampedMouseX - cropLeft, cropPreviewImg.naturalWidth - cropLeft));
    } else if (resizeHandle.includes("w")) {
      const newRight = cropLeft + cropWidth;
      newLeft = Math.max(0, Math.min(clampedMouseX, newRight - minSize));
      newWidth = newRight - newLeft;
    }

    if (resizeHandle.includes("s")) {
      newHeight = Math.max(minSize, Math.min(clampedMouseY - cropTop, cropPreviewImg.naturalHeight - cropTop));
    } else if (resizeHandle.includes("n")) {
      const newBottom = cropTop + cropHeight;
      newTop = Math.max(0, Math.min(clampedMouseY, newBottom - minSize));
      newHeight = newBottom - newTop;
    }

    cropLeft = newLeft;
    cropTop = newTop;
    cropWidth = newWidth;
    cropHeight = newHeight;
    updateCropBox();
    e.preventDefault();
  } else if (isDragging) {
    // Handle drag - clamp to image bounds
    cropLeft = Math.max(0, Math.min(clampedMouseX - dragOffsetX, cropPreviewImg.naturalWidth - cropWidth));
    cropTop = Math.max(0, Math.min(clampedMouseY - dragOffsetY, cropPreviewImg.naturalHeight - cropHeight));
    updateCropBox();
    e.preventDefault();
  } else if (isCropping) {
    // New crop selection
    cropLeft = Math.min(cropStartX, clampedMouseX);
    cropTop = Math.min(cropStartY, clampedMouseY);
    cropWidth = Math.abs(clampedMouseX - cropStartX);
    cropHeight = Math.abs(clampedMouseY - cropStartY);
    updateCropBox();
  }
});

document.addEventListener("mouseup", () => {
  isCropping = false;
  isDragging = false;
  isResizing = false;
});

// Close crop modal
btnCloseCropModal.addEventListener("click", () => {
  cropModal.style.display = "none";
  objectFileData = null;
});

// Back to upload
btnCropBack.addEventListener("click", () => {
  cropModal.style.display = "none";
  objectFileData = null;
});

// Clear crop selection
btnCropClear.addEventListener("click", () => {
  cropBox.style.display = "none";
  cropBox.innerHTML = "";
  btnCropSearch.disabled = true;
  cropLeft = 0;
  cropTop = 0;
  cropWidth = 0;
  cropHeight = 0;
});

// Crop search - crop the region and search
btnCropSearch.addEventListener("click", async () => {
  try {
    if (!objectFileData) {
      console.error("[CROP] No image data");
      return;
    }

    const rect = cropPreviewImg.getBoundingClientRect();
    const naturalWidth = cropPreviewImg.naturalWidth;
    const naturalHeight = cropPreviewImg.naturalHeight;

    // Display scaling ratio
    const scaleX = naturalWidth / rect.width;
    const scaleY = naturalHeight / rect.height;

    // object-fit: contain offset calculation
    const displayedWidth = (naturalWidth / naturalHeight) * rect.height;
    const displayedHeight = (naturalHeight / naturalWidth) * rect.width;
    const offsetX = rect.width > displayedWidth ? (rect.width - displayedWidth) / 2 : 0;
    const offsetY = rect.height > displayedHeight ? (rect.height - displayedHeight) / 2 : 0;

    // Get cropBox style values (display coords)
    let left = parseFloat(cropBox.style.left);
    let top = parseFloat(cropBox.style.top);
    let width = parseFloat(cropBox.style.width);
    let height = parseFloat(cropBox.style.height);

    // Convert to original image coordinates
    let x = Math.round((left - offsetX) * scaleX);
    let y = Math.round((top - offsetY) * scaleY);
    let w = Math.round(width * scaleX);
    let h = Math.round(height * scaleY);

    // Boundary clamp
    x = Math.max(0, Math.min(x, naturalWidth - 1));
    y = Math.max(0, Math.min(y, naturalHeight - 1));
    w = Math.max(32, Math.min(w, naturalWidth - x));
    h = Math.max(32, Math.min(h, naturalHeight - y));

    console.log(`[CROP] Calculated crop region: x=${x}, y=${y}, w=${w}, h=${h}`);

    // Close modal first
    cropModal.style.display = "none";

    // Set preview using canvas crop on frontend
    const img = new Image();
    const imgUrl = URL.createObjectURL(new Blob([objectFileData]));
    img.src = imgUrl;

    await new Promise<void>((resolve, reject) => {
      img.onload = async () => {
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d")!;
        ctx.drawImage(img, x, y, w, h, 0, 0, w, h);

        const blob = await new Promise<Blob>((res, rej) => {
          canvas.toBlob((b) => {
            if (!b) rej(new Error("Failed to create blob"));
            else res(b);
          }, "image/png");
        });

        // Set preview image
        const previewUrl = URL.createObjectURL(blob);
        objectPreviewImg.src = previewUrl;
        objectPreview.style.display = "flex";
        objectUpload.style.display = "none";
        objectRoisInfo.textContent = `已选择区域: ${w}×${h}`;
        resolve();
      };
      img.onerror = () => {
        console.error("[CROP] Image load error");
        reject(new Error("Failed to load image"));
      };
    });

    // Call Rust crop command with original image data
    const cropPath = await invoke<string>("write_cropped_image", {
      data: Array.from(objectFileData),
      mimeType: "image/png",
      x,
      y,
      w,
      h
    });

    objectImagePath = cropPath;
    btnObjectSearch.disabled = false;
    console.log("[CROP] Crop saved at:", cropPath);
  } catch (err) {
    console.error("[CROP] Failed to save crop:", err);
    objectRoisInfo.textContent = "保存裁剪失败: " + err;
  }
});

btnObjectClear.addEventListener("click", () => {
  objectPreview.style.display = "none";
  objectUpload.style.display = "flex";
  objectImagePath = null;
  objectFileData = null;
  objectResults.innerHTML = "";
  btnObjectSearch.disabled = true;
  objectRoisInfo.textContent = "";
  cropModal.style.display = "none";
  selectedObjectResults.clear();
  document.getElementById("object-export-controls")!.style.display = 'none';
  document.getElementById("object-pagination")!.innerHTML = '';
});

// ObjectSearch 搜索按钮
if (btnObjectSearch) {
  btnObjectSearch.addEventListener("click", async () => {
    if (!objectImagePath) return;
    objectResults.innerHTML = '<p style="color: var(--text-secondary)">搜索中...</p>';
    btnObjectSearch.disabled = true;
    try {
      const results = await invoke<any[]>("search_objects", { queryImage: objectImagePath, topK: 20 });
      if (results.length === 0) {
        objectResults.innerHTML = '<p style="color: var(--warning)">未找到相似物品</p>';
        document.getElementById("object-pagination")!.innerHTML = '';
        document.getElementById("object-export-controls")!.style.display = 'none';
      } else {
        currentObjectResults = results;
        objectTotalResults = results.length;
        objectPageNum = 1;
        selectedObjectResults.clear();
        updateObjectExportControls();
        document.getElementById("object-export-controls")!.style.display = 'flex';
        await renderObjectResultsPage();
      }
    } catch (e) {
      objectResults.innerHTML = `<p style="color: var(--danger)">搜索失败: ${e}</p>`;
    }
    btnObjectSearch.disabled = false;
  });
}

async function renderObjectResultsPage() {
  const objectResults = document.getElementById("object-results")!;
  const objectPagination = document.getElementById("object-pagination")!;
  const RESULTS_PER_PAGE = 20;

  const startIdx = (objectPageNum - 1) * RESULTS_PER_PAGE;
  const endIdx = Math.min(startIdx + RESULTS_PER_PAGE, objectTotalResults);
  const pageResults = currentObjectResults.slice(startIdx, endIdx);

  const htmlParts = [];
  for (let i = 0; i < pageResults.length; i++) {
    const r = pageResults[i];
    const globalIdx = startIdx + i;
    const displayName = r.image_name || (r.image_path ? r.image_path.split('/').pop() : `Image #${r.image_id}`);
    const truncatedName = displayName.length > 20 ? displayName.slice(0, 17) + '...' : displayName;
    const similarity = r.similarity || r.confidence || 0;

    let imgTag = '';
    if (r.thumbnail_path) {
      try {
        const base64 = await Promise.race([
          invoke<string>("get_image_thumbnail", { imagePath: r.thumbnail_path }),
          new Promise((_, reject) => setTimeout(() => reject(new Error("Timeout")), 5000))
        ]);
        imgTag = `<img src="${base64}" alt="${truncatedName}" style="width:100%;aspect-ratio:1;object-fit:cover;" />`;
      } catch (e) {
        imgTag = `<div style="background:linear-gradient(135deg,#667eea,#764ba2);display:flex;align-items:center;justify-content:center;color:white;aspect-ratio:1;font-size:12px;">ERR</div>`;
      }
    } else {
      imgTag = `<div style="background:linear-gradient(135deg,#667eea,#764ba2);display:flex;align-items:center;justify-content:center;color:white;aspect-ratio:1;font-size:12px;">${truncatedName}</div>`;
    }

    const isSelected = selectedObjectResults.has(globalIdx);
    htmlParts.push(`
      <div class="result-card" data-index="${globalIdx}" style="cursor:pointer;">
        <div class="checkbox-overlay ${isSelected ? 'checked' : ''}" data-result-index="${globalIdx}">
          ${isSelected ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"></polyline></svg>' : ''}
        </div>
        ${imgTag}
        <div class="result-info">
          <div class="result-filename" title="${r.image_path || ''}">${truncatedName}</div>
          <div class="result-similarity" style="font-size:11px;color:${similarity >= 0.7 ? '#4ade80' : similarity >= 0.5 ? '#facc15' : '#f87171'};">
            相似度: ${(similarity * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    `);
  }
  objectResults.innerHTML = htmlParts.join('');

  // Render pagination
  const totalPages = Math.ceil(objectTotalResults / RESULTS_PER_PAGE);
  let paginationHtml = '';
  if (totalPages > 1) {
    paginationHtml = `
      <button class="page-btn" data-page="${objectPageNum - 1}" ${objectPageNum === 1 ? 'disabled' : ''}>上一页</button>
      <span class="page-info">第 ${objectPageNum} / ${totalPages} 页</span>
      <button class="page-btn" data-page="${objectPageNum + 1}" ${objectPageNum === totalPages ? 'disabled' : ''}>下一页</button>
    `;
  }
  objectPagination.innerHTML = paginationHtml;
}

// Event delegation for object result card clicks
objectResults.addEventListener("click", (e) => {
  const target = e.target as Element;
  const card = target.closest('.result-card');
  if (card) {
    const indexStr = card.getAttribute('data-index');
    if (indexStr !== null) {
      const idx = parseInt(indexStr, 10);
      if (selectedObjectResults.has(idx)) {
        selectedObjectResults.delete(idx);
      } else {
        selectedObjectResults.add(idx);
      }
      updateObjectExportControls();
      renderObjectResultsPage();
    }
  }
});

// Event delegation for object pagination clicks
document.getElementById("object-pagination")?.addEventListener("click", (e) => {
  const btn = (e.target as HTMLElement).closest('.page-btn');
  if (btn && !(btn as HTMLButtonElement).disabled) {
    const page = parseInt((btn as HTMLElement).dataset.page!, 10);
    const totalPages = Math.ceil(objectTotalResults / 20);
    if (page >= 1 && page <= totalPages) {
      objectPageNum = page;
      renderObjectResultsPage();
    }
  }
});

// Object search export controls
const btnObjectChooseDir = document.getElementById("btn-object-choose-dir") as HTMLButtonElement;
const btnObjectExport = document.getElementById("btn-object-export") as HTMLButtonElement;
const objectSelectAll = document.getElementById("object-select-all") as HTMLInputElement;

function updateObjectExportControls() {
  const count = selectedObjectResults.size;
  const countEl = document.getElementById("object-export-count");
  if (countEl) countEl.textContent = `已选择: ${count}`;
  if (btnObjectExport) btnObjectExport.disabled = count === 0 || !objectExportDir;
}

btnObjectChooseDir?.addEventListener("click", async () => {
  try {
    const selected = await open({ directory: true });
    if (selected) {
      objectExportDir = selected as string;
      document.getElementById("object-export-dir-path")!.textContent = objectExportDir;
      updateObjectExportControls();
    }
  } catch (e) {
    console.error("选择导出目录失败:", e);
  }
});

btnObjectExport?.addEventListener("click", async () => {
  if (selectedObjectResults.size === 0 || !objectExportDir) return;
  const selectedPaths: string[] = [];
  for (const idx of selectedObjectResults) {
    if (currentObjectResults[idx]?.thumbnail_path) {
      selectedPaths.push(currentObjectResults[idx].thumbnail_path);
    }
  }
  if (selectedPaths.length > 0) {
    try {
      const results = await invoke<string[]>("copy_files", { sourcePaths: selectedPaths, destDir: objectExportDir });
      alert(`成功复制 ${results.length} 个文件到: ${objectExportDir}`);
      console.log("[EXPORT] Copied files:", results);
    } catch (e) {
      alert(`复制失败: ${e}`);
      console.error("[EXPORT] Copy failed:", e);
    }
  }
});

objectSelectAll?.addEventListener("change", () => {
  if (objectSelectAll.checked) {
    const totalPages = Math.ceil(objectTotalResults / 20);
    for (let p = 1; p <= totalPages; p++) {
      const start = (p - 1) * 20;
      const end = Math.min(start + 20, objectTotalResults);
      for (let i = start; i < end; i++) {
        selectedObjectResults.add(i);
      }
    }
  } else {
    selectedObjectResults.clear();
  }
  updateObjectExportControls();
  renderObjectResultsPage();
});

// Init
window.addEventListener("DOMContentLoaded", () => {
  loadStatistics();
  setInterval(loadStatistics, 5000);
});