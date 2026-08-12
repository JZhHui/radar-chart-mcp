import os
import io
import base64
import urllib.request
from typing import List, Dict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse, PlainTextResponse
import uvicorn


def setup_chinese_font():
    font_path = "/tmp/NotoSansSC-Regular.otf"
    if not os.path.exists(font_path):
        urls = [
            "https://cdn.jsdelivr.net/gh/googlefonts/noto-cjk@main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf",
            "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf",
        ]
        for url in urls:
            try:
                urllib.request.urlretrieve(url, font_path)
                break
            except Exception:
                continue
    if os.path.exists(font_path):
        try:
            from matplotlib import font_manager
            font_manager.fontManager.addfont(font_path)
            prop = FontProperties(fname=font_path)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return prop
        except Exception as e:
            print(f"[WARN] 字体加载失败: {e}")
    plt.rcParams['font.sans-serif'] = [
        'Noto Sans CJK SC', 'SimHei', 'WenQuanYi Micro Hei',
        'DejaVu Sans', 'Arial Unicode MS', 'sans-serif'
    ]
    plt.rcParams['axes.unicode_minus'] = False
    return None


CHINESE_FONT = setup_chinese_font()
mcp = FastMCP("CareerRadarMCP")


@mcp.tool()
def draw_career_radar_chart(
        abilities: List[Dict[str, int]],
        user_name: str = "用户",
        chart_title: str = "职业能力雷达图"
) -> str:
    if not abilities or len(abilities) < 3:
        return "❌ 参数错误：请至少提供 3 个能力维度（建议提供 6 项）"

    categories = [a.get("name", f"维度{i + 1}") for i, a in enumerate(abilities)]
    scores = [min(100, max(0, a.get("score", 0))) for a in abilities]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    scores_closed = scores + scores[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    ax.fill(angles_closed, scores_closed, color='#6366F1', alpha=0.25)
    ax.plot(angles_closed, scores_closed, color='#4F46E5', linewidth=3, marker='o', markersize=10)
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=14, fontproperties=CHINESE_FONT)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10, color='#9CA3AF')
    ax.grid(True, linestyle='--', alpha=0.5, color='#D1D5DB')

    title_text = f"{user_name} - {chart_title}"
    ax.set_title(title_text, fontsize=18, fontweight='bold', pad=25,
                 fontproperties=CHINESE_FONT, color='#111827')

    for angle, score in zip(angles, scores):
        offset = 10 if score < 92 else -14
        ax.text(angle, score + offset, str(score),
                ha='center', va='center', fontsize=12,
                fontweight='bold', color='#1F2937',
                fontproperties=CHINESE_FONT)

    ax.text(0, -0.12, "💡 分数范围 0-100 | 越靠近外圈代表能力越强",
            transform=ax.transAxes, fontsize=10, color='#6B7280',
            ha='center', fontproperties=CHINESE_FONT)

    try:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        return f"❌ 图片生成失败: {str(e)}"

    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    min_score = min(scores)
    max_ability = categories[scores.index(max_score)]
    min_ability = categories[scores.index(min_score)]

    if avg_score >= 80:
        direction = "管理型/专家型"
        advice = "你的综合能力非常突出，具备向高级管理岗或技术专家方向发展的潜力。"
    elif avg_score >= 65:
        direction = "骨干型"
        advice = "你的基础能力扎实，是团队中的中坚力量。建议向「T型人才」发展。"
    else:
        direction = "成长型"
        advice = "你处于职业成长初期，建议聚焦 1-2 个核心能力方向重点突破。"

    report = f"""## 🎯 {user_name} 的职业能力分析报告

![{chart_title}](data:image/png;base64,{img_base64})

### 📊 能力评分详情
| 能力维度 | 分数 | 评级 |
|---------|------|------|
"""
    for cat, score in zip(categories, scores):
        level = "优秀" if score >= 80 else "良好" if score >= 60 else "待提升"
        report += f"| {cat} | {score}分 | {level} |\n"

    report += f"""
### 💡 核心洞察
- **综合评分**: {avg_score:.1f} / 100
- **核心优势**: **{max_ability}**（{max_score}分）
- **提升空间**: **{min_ability}**（{min_score}分）

### 🚀 职业发展方向：{direction}
{advice}
"""
    return report


@mcp.tool()
def analyze_career_path(
        current_role: str,
        target_role: str,
        abilities: List[Dict[str, int]]
) -> str:
    role_requirements = {
        "技术总监": {"技术能力": 90, "领导力": 85, "沟通能力": 80, "创新能力": 75, "执行力": 85, "学习能力": 80},
        "产品经理": {"技术能力": 60, "领导力": 70, "沟通能力": 90, "创新能力": 85, "执行力": 80, "学习能力": 75},
        "项目经理": {"技术能力": 65, "领导力": 80, "沟通能力": 90, "创新能力": 60, "执行力": 90, "学习能力": 70},
        "架构师": {"技术能力": 95, "领导力": 70, "沟通能力": 75, "创新能力": 85, "执行力": 70, "学习能力": 90},
        "团队主管": {"技术能力": 75, "领导力": 85, "沟通能力": 85, "创新能力": 70, "执行力": 80, "学习能力": 75},
    }

    req = role_requirements.get(target_role)
    if not req:
        return f"❌ 暂不支持'{target_role}'的路径分析。目前支持: {', '.join(role_requirements.keys())}"

    current_map = {a["name"]: a["score"] for a in abilities}
    gaps = []
    for skill, required in req.items():
        actual = current_map.get(skill, 0)
        gap = required - actual
        if gap > 0:
            gaps.append((skill, gap, required, actual))

    gaps.sort(key=lambda x: x[1], reverse=True)

    report = f"""## 🛤️ {current_role} → {target_role} 职业路径分析

### 📍 能力差距矩阵
| 能力维度 | 当前 | 目标 | 差距 | 优先级 |
|---------|------|------|------|--------|
"""
    for skill, gap, required, actual in gaps[:6]:
        priority = "🔴 紧急" if gap > 20 else "🟡 重要" if gap > 10 else "🟢 次要"
        report += f"| {skill} | {actual} | {required} | +{gap} | {priority} |\n"

    if not gaps:
        report += "\n🎉 恭喜！你已经具备目标岗位的核心能力要求！\n"
    else:
        top = gaps[0]
        report += f"\n### 🎯 优先提升策略\n**关键短板**: {top[0]}（差距 {top[1]} 分）\n"

    return report


async def health_check(request):
    return JSONResponse({"status": "ok", "service": "CareerRadarMCP", "version": "1.0.0"})


async def root_page(request):
    return PlainTextResponse(
        "Career Radar MCP Server is running.\nSSE Endpoint: /sse\n"
    )


app = Starlette(routes=[
    Route("/", root_page),
    Route("/health", health_check),
    Mount("/", app=mcp.sse_app()),
])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"[INFO] Starting CareerRadarMCP on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
