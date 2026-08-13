import os
import io
import asyncio
import hashlib
import json
from datetime import datetime
from typing import List
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.interpolate import make_interp_spline
from PIL import Image

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.staticfiles import StaticFiles
import aiofiles
import uvicorn

# 确保在 app 初始化或首次调用时执行
STATIC_DIR = "/tmp/mcp_radar_charts"
os.makedirs(STATIC_DIR, exist_ok=True)
# ==================== 配置区 ====================
PORT = int(os.environ.get("PORT", 8080))
HOST = os.environ.get("HOST", "0.0.0.0")
# 自动检测 Railway 公网域名
railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
if railway_domain:
    BASE_URL = f"https://{railway_domain}"
else:
    BASE_URL = os.environ.get("BASE_URL", f"http://localhost:{PORT}")
STATIC_DIR = "/tmp/mcp_radar_charts"
FONT_PATH = "/tmp/NotoSansSC-Regular.otf"

os.makedirs(STATIC_DIR, exist_ok=True)
_plot_executor = ThreadPoolExecutor(max_workers=2)

# ==================== 字体与工具函数 ====================
@lru_cache(maxsize=1)
def setup_chinese_font():
    """懒加载 + 缓存中文字体"""
    if not os.path.exists(FONT_PATH):
        import urllib.request
        urls = [
            "https://cdn.jsdelivr.net/gh/googlefonts/noto-cjk@main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf",
            "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf",
        ]
        for url in urls:
            try:
                urllib.request.urlretrieve(url, FONT_PATH)
                break
            except Exception:
                continue

    if os.path.exists(FONT_PATH):
        try:
            from matplotlib import font_manager
            font_manager.fontManager.addfont(FONT_PATH)
            prop = FontProperties(fname=FONT_PATH)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return prop
        except Exception as e:
            print(f"[WARN] 字体加载失败: {e}")

    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return None


def smooth_polar_data(angles, scores, num_points=300):
    """周期性边界条件的贝塞尔平滑，避免首尾折角"""
    angles_closed = np.concatenate([angles, [angles[0] + 2 * np.pi]])
    scores_closed = np.concatenate([scores, [scores[0]]])
    angles_ext = np.concatenate([[angles_closed[-2] - 2 * np.pi], angles_closed, [angles_closed[1] + 2 * np.pi]])
    scores_ext = np.concatenate([[scores_closed[-2]], scores_closed, [scores_closed[1]]])

    angles_smooth = np.linspace(0, 2 * np.pi, num_points)
    spline = make_interp_spline(angles_ext, scores_ext, k=3)
    scores_smooth = spline(angles_smooth)
    return angles_smooth, scores_smooth


def _generate_radar_sync(categories, scores, user_name, chart_title, chinese_font):
    """纯同步绘图函数，在线程池中执行，不阻塞异步事件循环"""
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_smooth, scores_smooth = smooth_polar_data(np.array(angles), np.array(scores))

    avg = sum(scores) / len(scores)
    if avg >= 80:
        line_color = '#10B981'  # 绿色（优秀）
    elif avg >= 60:
        line_color = "#5C77DA"  # 紫色（良好）
    else:
        line_color = "#DAA64B"

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')

    ax.fill(angles_smooth, scores_smooth, color=fill_color, alpha=0.2)
    ax.plot(angles_smooth, scores_smooth, color=line_color, linewidth=3)
    ax.scatter(angles, scores, color=line_color, s=80, zorder=5)

    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=13, fontproperties=chinese_font)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9, color='#9CA3AF')
    ax.grid(True, linestyle='--', alpha=0.5, color='#D1D5DB')

    ax.set_title(f"{user_name} - {chart_title}", fontsize=17, fontweight='bold', pad=25,
                 fontproperties=chinese_font, color='#111827')

    for angle, score in zip(angles, scores):
        offset = 10 if score < 92 else -14
        ax.text(angle, score + offset, str(int(score)), ha='center', va='center',
                fontsize=11, fontweight='bold', color='#1F2937', fontproperties=chinese_font)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)

    # Pillow 二次压缩 JPEG
    img = Image.open(buf).convert('RGB')
    compressed = io.BytesIO()
    img.save(compressed, format='JPEG', quality=75, optimize=True)
    return compressed.getvalue()


# ==================== Pydantic 入参校验 ====================
class AbilityItem(BaseModel):
    name: str = Field(..., min_length=1, max_length=20, description="能力维度名称")
    score: float = Field(..., ge=0, le=100, description="分数 0-100")


# ==================== MCP Server 定义 ====================
mcp = FastMCP("CareerRadarMCP")


@mcp.tool()
async def draw_career_radar_chart(
    abilities: List[AbilityItem],
    user_name: str = "用户",
    chart_title: str = "职业能力雷达图"
) -> str:
    """
       生成职业能力雷达图。当用户提到技能掌握程度、能力评估、职业画像时自动调用。

       ⚠️ 调用规则（必须严格遵守）：
       1. 至少需要 3 个能力维度，若用户提供的不足 3 个，你必须根据用户的职业/岗位自动补全至 6 个维度
       2. 将用户的模糊描述转换为 0-100 的分数：
          - "精通/专家级" → 90-100
          - "熟练/比较熟练" → 75-89
          - "熟悉/一般" → 60-74
          - "了解/入门" → 40-59
          - "不熟练/不会" → 0-39
       3. 如果用户未提供具体维度名称，根据其提到的职业自动推断常见能力维度
       4. user_name 从对话上下文中提取，提取不到则使用"用户"
       """
    if len(abilities) < 3:
        return "❌ 参数错误：请至少提供 3 个能力维度（建议 6 项）"

    categories = [a.name for a in abilities]
    scores = [a.score for a in abilities]
    chinese_font = setup_chinese_font()

    # ✅ 核心：绘图放入线程池，不阻塞 Starlette 异步事件循环
    loop = asyncio.get_running_loop()
    image_bytes = await loop.run_in_executor(
        _plot_executor,
        _generate_radar_sync,
        categories, scores, user_name, chart_title, chinese_font
    )

    # ✅ 以内容哈希命名，相同数据命中缓存，避免重复生成
    content_hash = hashlib.md5(image_bytes).hexdigest()[:12]
    filename = f"radar_{content_hash}.jpg"
    filepath = os.path.join(STATIC_DIR, filename)

    async with aiofiles.open(filepath, 'wb') as f:
        await f.write(image_bytes)

    image_url = f"{BASE_URL}/static/{filename}"

    # ===== 分析算法（含偏科指数）=====
    avg_score = sum(scores) / len(scores)
    std_dev = float(np.std(scores))
    cv = std_dev / avg_score if avg_score > 0 else 0
    max_idx, min_idx = scores.index(max(scores)), scores.index(min(scores))

    if cv > 0.3:
        balance = "⚠️ 严重偏科"
        balance_advice = "能力呈现明显长板效应，短板可能成为晋升瓶颈，建议优先补齐低于60分的维度。"
    elif cv > 0.15:
        balance = "⚖️ 轻度偏科"
        balance_advice = "能力分布不均，建议保持优势的同时分配20%精力提升弱势维度。"
    else:
        balance = "🌟 均衡发展"
        balance_advice = "能力模型健康均衡，具备复合型管理人才潜质。"

    if avg_score >= 80:
        direction, advice = "管理型/专家型", "综合能力突出，具备向高级管理岗或技术专家方向发展的潜力。"
    elif avg_score >= 65:
        direction, advice = "骨干型", "基础能力扎实，是团队中坚力量，建议向「T型人才」发展。"
    else:
        direction, advice = "成长型", "处于职业成长初期，建议聚焦1-2个核心能力重点突破。"

    table_rows = "\n".join(
        f"| {c} | {int(s)}分 | {'优秀' if s >= 80 else '良好' if s >= 60 else '待提升'} |"
        for c, s in zip(categories, scores)
    )

    return f"""## 🎯 {user_name} 的职业能力分析报告

![{chart_title}]({image_url})

### 📊 能力评分详情

| 能力维度 | 分数 | 评级 |
|---------|------|------|
{table_rows}

### 💡 核心洞察
- **综合评分**: {avg_score:.1f} / 100
- **能力健康度**: {balance}
- **核心优势**: **{categories[max_idx]}**（{int(scores[max_idx])}分）
- **提升空间**: **{categories[min_idx]}**（{int(scores[min_idx])}分）

### ⚖️ 均衡度建议
{balance_advice}

### 🚀 职业发展方向：{direction}
{advice}
"""


@mcp.tool()
def analyze_career_path(
    current_role: str,
    target_role: str,
    abilities: List[AbilityItem]
) -> str:
    """分析当前岗位到目标岗位的能力差距"""
    ROLE_REQUIREMENTS = {
        "技术总监": {"技术能力": 90, "领导力": 85, "沟通能力": 80, "创新能力": 75, "执行力": 85, "学习能力": 80},
        "产品经理": {"技术能力": 60, "领导力": 70, "沟通能力": 90, "创新能力": 85, "执行力": 80, "学习能力": 75},
        "项目经理": {"技术能力": 65, "领导力": 80, "沟通能力": 90, "创新能力": 60, "执行力": 90, "学习能力": 70},
        "架构师":   {"技术能力": 95, "领导力": 70, "沟通能力": 75, "创新能力": 85, "执行力": 70, "学习能力": 90},
        "团队主管": {"技术能力": 75, "领导力": 85, "沟通能力": 85, "创新能力": 70, "执行力": 80, "学习能力": 75},
    }

    req = ROLE_REQUIREMENTS.get(target_role)
    if not req:
        return f"❌ 暂不支持'{target_role}'，目前支持: {', '.join(ROLE_REQUIREMENTS.keys())}"

    current_map = {a.name: a.score for a in abilities}
    gaps = sorted(
        [(skill, req[skill] - current_map.get(skill, 0), req[skill], current_map.get(skill, 0))
         for skill in req if req[skill] - current_map.get(skill, 0) > 0],
        key=lambda x: x[1], reverse=True
    )

    report = f"## 🛤️ {current_role} → {target_role} 路径分析\n\n### 📍 能力差距矩阵\n| 能力 | 当前 | 目标 | 差距 | 优先级 |\n|------|------|------|------|--------|\n"
    for skill, gap, required, actual in gaps[:6]:
        priority = "🔴 紧急" if gap > 20 else "🟡 重要" if gap > 10 else "🟢 次要"
        report += f"| {skill} | {int(actual)} | {required} | +{int(gap)} | {priority} |\n"

    if not gaps:
        report += "\n🎉 恭喜！你已具备目标岗位的核心能力要求！\n"
    else:
        report += f"\n### 🎯 优先提升: **{gaps[0][0]}**（差距 {int(gaps[0][1])} 分）\n"
    return report


# ==================== Starlette App 组装 ====================
async def health_check(request):
    return JSONResponse({"status": "ok", "service": "CareerRadarMCP"})

async def root_page(request):
    return PlainTextResponse(f"CareerRadarMCP running.\nSSE: {BASE_URL}/sse\nStatic: {BASE_URL}/static/\n")

app = Starlette(routes=[
    Route("/", root_page),
    Route("/health", health_check),
    Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),  # ✅ 图片URL托管
    Mount("/", app=mcp.sse_app()),                                   # ✅ SSE 独立挂载
])

if __name__ == "__main__":
    print(f"[INFO] Starting CareerRadarMCP on {HOST}:{PORT}")
    print(f"[INFO] Image URLs will be served at {BASE_URL}/static/")
    uvicorn.run(app, host=HOST, port=PORT)
