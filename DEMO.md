# metricpower Shiny app — 本地安裝與示範教程

適用：macOS（Windows/Linux 差異見最後）。整個流程約 5 分鐘；只需要 Python 3.10 以上，不需要 R。

---

## 1. 安裝（只做一次）

### 1.1 確認 Python 版本

```bash
python3 --version        # 需要 3.10 以上；3.11 最穩
```

如果是 3.8 / 3.9（macOS 內建的那個），先裝新的：

```bash
brew install python@3.11
python3.11 --version
```

（之後所有 `python3` 都換成 `python3.11`。）

### 1.2 解壓縮並建立虛擬環境

把 `metricpower-py.zip` 解壓到你要放的地方（例如 `~/Desktop/Ronnie/`），然後：

```bash
cd ~/Desktop/Ronnie/metricpower-py      # 進到解壓後的資料夾（裡面要看得到 app.py 和 pyproject.toml）
python3 -m venv .venv                   # 建一個獨立的虛擬環境，不碰系統 Python
source .venv/bin/activate               # 啟用；提示字元前面會出現 (.venv)
pip install --upgrade pip
pip install -e ".[app]"                 # 裝 metricpower + Shiny（約 1 分鐘）
```

### 1.3 確認裝好了

```bash
python -c "import metricpower; print(metricpower.__version__)"   # 0.1.1
python -m pytest -q                                              # 選擇性：74 passed
```

---

## 2. 開 app

```bash
cd ~/Desktop/Ronnie/metricpower-py
source .venv/bin/activate              # 每次開新 terminal 都要先做這步
shiny run app.py
```

終端機會印 `Uvicorn running on http://127.0.0.1:8000`。用瀏覽器開 **http://127.0.0.1:8000**。

- 關掉：回到終端機按 **Ctrl + C**。
- 8000 被占用：`shiny run app.py --port 8001`，網址改成 8001。
- 改了程式碼想即時看到：`shiny run app.py --reload`。

---

## 3. 示範腳本（約 8 分鐘，四個場景）

畫面分左右：**左 = 輸入**（metric、H1 的 operating point、H0、alpha、power、clustering），**右 = 這個數字怎麼算的**（一般公式 → 代入數字 → 每一步的中間量 → 注意事項 → 文獻）。

### 場景 A：Dice coefficient（預設畫面）— 展示整個流程

1. 打開時預設就是 **Dice / F1 (pooled)**，sensitivity 0.7、specificity 0.8、prevalence 0.5，H0 = 0.6868（比 H1 低 0.05）。
2. 指右邊：`θ = 2TP/(2TP+FP+FN)` → `Σ = diag(p) − pp′, V = ∇g′Σ∇g` → `n = (1.959964 + 0.841621)² × V / Δ²`。z 值是六位小數，不是 1.96。
3. 指 **test statistic** 那一排：**score — recommended** 是粗體，下面一行寫為什麼（模擬顯示 Wald n 的 power 只有 0.77–0.79，score n 是 0.79–0.81），再下面是 disclaimer。切到 Wald，n 從 920 變 809，右邊公式同步換成 Wald 版。
4. 把 prevalence 改成 0.05：n 跳到幾千。說明：低盛行率下 Dice 本身掉到 0.25，同樣的 Δ = 0.05 相對變大，V 也變大。改回 0.5。

### 場景 B：Metric 下拉選單 — 展示分類

1. 點 metric 下拉：灰色標題是 11 個 problem category（confusion-matrix classification、precision–recall、ROC/AUC、calibration、segmentation、distance、regression、correlation/agreement、information theory、fairness/survival/decision/NLP、image quality）。
2. 指出同一個 metric 可以出現在兩個類別下（Dice 在 1 和 5），選哪個結果一樣。
3. 選 **AUROC**：輸入變成 AUC₁ / AUC₀ / 兩組比例，右邊是 Hanley–McNeil 的 variance 公式與文獻（這是「已有軟體」的 metric，原生實作、不需要裝 R 套件）。

### 場景 C：Per-case metric — 展示 t-test 修正

1. 選 **brier**（Brier score, per-case mean）。輸入是 pilot 的 mean 和 sd（預設 0.80 / 0.12，H0 = 0.77 → n = 128）。
2. 右邊：`Var(x̄) = σ²/n`，然後 **n 用 noncentral t 算**，steps 裡同時列 `n (z formula)` 和 `n (t-test)`。
3. 把 H0 改成 0.70（Δ 變大，n 掉到 20 以下）：note 出現「n uses the t-test rather than the z formula」。說明：小樣本用 sample SD 當 SE，z 的臨界值會讓 type I error 膨脹（n = 5 時 0.12），t 修正回 0.05。

### 場景 D：Volumetric similarity — 展示「delta method 失效時怎麼辦」

1. 選 **volumetric_similarity**，用預設 operating point。
2. 右邊 note：`|fp − fn| is 1.6 SE from the kink at fp = fn … n is the sample size of the folded-normal test`。說明：這個 metric 有絕對值，在 fp = fn 不可微；app 自動換成 folded-normal 的精確計算，並告訴你 delta method 的 n 是多少、point estimate 的 bias 是多少。
3. 把 specificity 改成 0.9（fp 遠離 fn）：note 裡的 SE 距離變大，n 和 delta method 的接近。

### 場景 E（選擇性）：Clustering

在任何 metric 下，把 **units per cluster m̄** 設 3、**intra-cluster correlation ρ** 設 0.3：下面顯示 `design effect = 1 + (3 − 1) × 0.3 = 1.6`，n 乘 1.6。說明：一個病人多個病灶時，病灶不是獨立樣本。

---

## 4. 示範時常被問的問題

| 問題 | 回答 |
|---|---|
| Sensitivity / specificity / prevalence 從哪來？ | 投稿前的 pilot、文獻中同類模型的表現、或你願意接受的最低表現。prevalence 是**你要收的測試集**裡的陽性比例，不是母體的。 |
| H0 要填什麼？ | 你想排除的值：要打敗的 benchmark、或可接受的最低表現。n 跟 1/Δ² 成正比，Δ 減半 n 變四倍。 |
| Wald 和 score 差在哪？ | 同一個 test，SE 一個用資料算（Wald）、一個用 H0 算（score）。分析時用哪個 SE，就用哪個 n。粗體是建議，不是規定。 |
| 這個數字可信嗎？ | 每個 metric 都用 10,000 次模擬驗證過 type I error 和 power（repo 的 `validation/operating_characteristics.txt`）。右邊的 "Read before trusting the number" 會列出該 metric 的警告（小 n、skew、sparse cell）。 |
| 為什麼有些 metric 不在清單裡？ | Hausdorff、average precision、ECE、BLEU 等沒有 analytic variance，無法 a priori 算 n，是 Part II 的範圍。 |

---

## 5. 常見錯誤

| 訊息 | 原因 / 解法 |
|---|---|
| `command not found: shiny` | 沒有啟用虛擬環境：`source .venv/bin/activate` |
| `Address already in use` | 8000 被上一次沒關的 app 占住：換 `--port 8001`，或 `lsof -i :8000` 找到 PID 後 `kill` |
| `ERROR: Package requires a different Python` | Python 太舊，用 `python3.11 -m venv .venv` 重建 |
| 公式顯示成原始 LaTeX（`\frac{...}`） | 重新整理頁面；MathJax 是內建的（`app/www/mathjax`），不需要網路 |
| 下拉選單被下面的框蓋住 | 舊版問題，已改為原生 select；若仍出現，確認 `metricpower.__version__` 是 0.1.1 |

---

## 6. Windows / Linux

- Windows：`py -3.11 -m venv .venv`，啟用用 `.venv\Scripts\activate`，其餘相同。
- Linux：和 macOS 相同；若沒有 `python3-venv`，先 `sudo apt install python3.11-venv`。

## 7. 不想裝也能看

把 app 部署到 shinyapps.io（免費方案夠用）之後，任何人開網址就能用：

```bash
pip install rsconnect-python
rsconnect add --account <帳號> --name <帳號> --token <token> --secret <secret>
rsconnect deploy shiny . --name <帳號> --title metricpower --entrypoint app:app
```

token / secret 在 shinyapps.io → Account → Tokens。
