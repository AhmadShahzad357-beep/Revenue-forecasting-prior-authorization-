"""Animated architecture diagrams for the README, replacing the two static
Mermaid diagrams with GIFs in the same visual language used across the
RCM-Intelligence project (consistent palette, pulse-travel animation).
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

OUT_DIR = "docs/assets"
PALETTE = {"blue": "#2E5EAA", "green": "#2E8B57", "orange": "#D98E04", "purple": "#6A4C93", "red": "#C0392B"}
DIM_EDGE = "#DDE1E7"; DIM_TEXT = "#A3AAB5"; ON_TEXT = "#1F2937"; BG_OFF = "#FBFCFD"; SHADOW = "#E9ECEF"

def ease_in_out(t): return t * t * (3 - 2 * t)

def tint(hex_color, alpha=0.14):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * (1 - alpha)); g = int(g + (255 - g) * (1 - alpha)); b = int(b + (255 - b) * (1 - alpha))
    return f"#{r:02X}{g:02X}{b:02X}"

STAGES_1 = [
    ("Data Sources", "CA IMR + NY DFS\nappeals, CMS payments", PALETTE["blue"]),
    ("Data Preparation", "clean, leakage checks,\nEDA, feature pipeline", PALETTE["orange"]),
    ("Modelling", "3 models, time-aware CV,\nper-State calibration", PALETTE["green"]),
    ("Payment Proxy", "expected vs actual,\nexposure index", PALETTE["purple"]),
    ("Serving", "FastAPI + dashboard,\none command to run", PALETTE["red"]),
]
CAPTIONS_1 = [
    "Loading California IMR, New York DFS, and CMS payment data...",
    "Cleaning, checking for leakage, and building the feature pipeline...",
    "Training 3 models with time-aware validation and per-State calibration...",
    "Converting calibrated probabilities into a payment-proxy exposure...",
    "Serving the real model through the API and dashboard.",
]

def build_row_figure(stages, title_text, figsize=(14, 4.4)):
    n = len(stages); x = [i * 2.1 for i in range(n)]; y = 0.0; box_w, box_h = 1.75, 1.0
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(-1.1, x[-1] + 1.1); ax.set_ylim(-1.9, 1.6); ax.axis("off")
    boxes = []
    for i, (t, s, color) in enumerate(stages):
        shadow = mpatches.FancyBboxPatch((x[i]-box_w/2+0.045, y-box_h/2-0.05), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=0, facecolor=SHADOW, zorder=2, alpha=0.7)
        ax.add_patch(shadow)
        glow = mpatches.FancyBboxPatch((x[i]-box_w/2-0.06, y-box_h/2-0.06), box_w+0.12, box_h+0.12,
            boxstyle="round,pad=0.02,rounding_size=0.16", linewidth=0, facecolor=color, zorder=2, alpha=0.0)
        ax.add_patch(glow)
        box = mpatches.FancyBboxPatch((x[i]-box_w/2, y-box_h/2), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.8, edgecolor=DIM_EDGE, facecolor=BG_OFF, zorder=3)
        ax.add_patch(box)
        title_txt = ax.text(x[i], y+0.18, t, ha="center", va="center", fontsize=10.5, color=DIM_TEXT, zorder=4)
        sub_txt = ax.text(x[i], y-0.28, s, ha="center", va="center", fontsize=7.6, color=DIM_TEXT, zorder=4)
        badge = ax.text(x[i], y+box_h/2+0.30, f"{i+1}", ha="center", va="center", fontsize=10, color="white",
            fontweight="bold", zorder=5, bbox=dict(boxstyle="circle,pad=0.34", facecolor=DIM_EDGE, edgecolor="none"))
        boxes.append(dict(box=box, glow=glow, title=title_txt, sub=sub_txt, badge=badge, color=color))
    lines = []
    for i in range(n-1):
        line, = ax.plot([x[i]+box_w/2, x[i+1]-box_w/2], [y, y], color=DIM_EDGE, linewidth=2.4, zorder=1, solid_capstyle="round")
        lines.append(line)
    trail = [ax.scatter([], [], s=0, color=PALETTE["blue"], zorder=6, linewidths=0) for _ in range(6)]
    pulse = ax.scatter([x[0]], [y], s=0, color=PALETTE["blue"], zorder=7, edgecolors="white", linewidths=1.4)
    caption = ax.text((x[0]+x[-1])/2, -1.55, "", ha="center", va="center", fontsize=11.2, color="#4B5563", style="italic")
    ax.text((x[0]+x[-1])/2, 1.35, title_text, ha="center", va="center", fontsize=13, color="#111827", fontweight="bold")
    fig.tight_layout()
    return fig, ax, boxes, lines, pulse, trail, caption, x, y, box_w

def animate_row(stages, captions, title_text, out_name, travel_frames=16, hold_frames=10):
    fig, ax, boxes, lines, pulse, trail, caption, x, y, box_w = build_row_figure(stages, title_text)
    n = len(stages); lit = set()
    def reset_all():
        lit.clear()
        for b in boxes:
            b["box"].set_edgecolor(DIM_EDGE); b["box"].set_linewidth(1.8); b["box"].set_facecolor(BG_OFF)
            b["glow"].set_alpha(0.0); b["title"].set_color(DIM_TEXT); b["title"].set_fontweight("normal")
            b["sub"].set_color(DIM_TEXT); b["badge"].get_bbox_patch().set_facecolor(DIM_EDGE)
        for line in lines: line.set_color(DIM_EDGE); line.set_linewidth(2.4)
        for t in trail: t.set_sizes([0])
    def light_up(i, pop_frame):
        lit.add(i); b = boxes[i]
        b["box"].set_edgecolor(b["color"]); b["box"].set_linewidth(2.8); b["box"].set_facecolor(tint(b["color"]))
        b["title"].set_color(ON_TEXT); b["title"].set_fontweight("bold"); b["sub"].set_color("#4B5563")
        b["badge"].get_bbox_patch().set_facecolor(b["color"])
        pop_t = min(pop_frame / hold_frames, 1.0)
        glow_alpha = (1 - abs(pop_t - 0.5) * 2) * 0.22
        b["glow"].set_alpha(max(glow_alpha, 0.05) if pop_t < 1.0 else 0.06)
    per_stage = travel_frames + hold_frames; total = per_stage * n + 22
    def update(frame):
        f = frame % (total + 14)
        if f == 0: reset_all(); caption.set_text(""); pulse.set_sizes([0])
        if f >= total:
            pulse.set_sizes([0])
            for t in trail: t.set_sizes([0])
            caption.set_text("\u2713 Case scored end to end.")
            return []
        stage_idx = min(f // per_stage, n - 1); within = f % per_stage
        if within < travel_frames and stage_idx > 0:
            t_lin = within / travel_frames; t = ease_in_out(t_lin)
            x0, x1 = x[stage_idx-1]+box_w/2, x[stage_idx]-box_w/2
            cx = x0 + (x1-x0)*t
            pulse.set_offsets([[cx, y]]); pulse.set_sizes([160])
            for k, tdot in enumerate(trail):
                lag = (within - (k+1)*2) / travel_frames
                if lag > 0:
                    tx = x0 + (x1-x0)*ease_in_out(min(lag,1.0))
                    tdot.set_offsets([[tx, y]]); tdot.set_sizes([90-k*12]); tdot.set_alpha(max(0.5-k*0.08, 0.05))
                else: tdot.set_sizes([0])
            lines[stage_idx-1].set_color(PALETTE["blue"]); lines[stage_idx-1].set_linewidth(3.2)
            caption.set_text(captions[stage_idx-1])
        else:
            pop_frame = within - travel_frames if within >= travel_frames else hold_frames
            light_up(stage_idx, pop_frame); pulse.set_sizes([0])
            for t in trail: t.set_sizes([0])
            caption.set_text(captions[stage_idx])
        return []
    anim = FuncAnimation(fig, update, frames=total+40, interval=90, blit=False)
    out_path = f"{OUT_DIR}/{out_name}"
    anim.save(out_path, writer=PillowWriter(fps=1000/90))
    plt.close(fig); print(f"Saved: {out_path}")

def animate_sequence():
    participants = ["Appeals\nTeam", "Dashboard", "FastAPI\napi.py", "Feature\nPipeline", "XGBoost\nModel", "Per-State\nCalibrator"]
    colors = [PALETTE["red"], PALETTE["red"], PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["green"]]
    n = len(participants); px = [i * 2.0 for i in range(n)]; top_y = 0.3; bottom_y = -6.6
    steps = [
        (0, 1, "Enter case (State, diagnosis, treatment, ...)"), (1, 2, "POST /api/predict"),
        (2, 3, "Encode with the SAME fitted pipeline used in training"), (3, 4, "37 features"),
        (4, 5, "Raw probability"), (5, 2, "Calibrated probability (per State)"),
        (2, 1, "Decision, proxy, context, warnings"), (1, 0, "Show the result"),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(-1.0, px[-1]+1.0); ax.set_ylim(bottom_y-0.6, top_y+1.1); ax.axis("off")
    ax.text((px[0]+px[-1])/2, top_y+0.85, "What Happens When One Case Is Scored", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#111827")
    lifelines = []
    for i, (name, color) in enumerate(zip(participants, colors)):
        box = mpatches.FancyBboxPatch((px[i]-0.75, top_y-0.28), 1.5, 0.56, boxstyle="round,pad=0.02,rounding_size=0.1",
            linewidth=1.8, edgecolor=color, facecolor=tint(color, 0.10), zorder=4)
        ax.add_patch(box)
        ax.text(px[i], top_y, name, ha="center", va="center", fontsize=9, color="#1F2937", fontweight="bold", zorder=5)
        line, = ax.plot([px[i], px[i]], [top_y-0.28, bottom_y], color=DIM_EDGE, linewidth=1.4, linestyle=(0,(4,3)), zorder=1)
        lifelines.append(line)
    step_y = [top_y - 0.85 - k * 0.82 for k in range(len(steps))]
    arrow_lines, arrow_labels, arrow_heads = [], [], []
    for k, (a, b, label) in enumerate(steps):
        y0 = step_y[k]
        line, = ax.plot([], [], color=DIM_EDGE, linewidth=2.2, zorder=2, solid_capstyle="round")
        arrow_lines.append(line)
        head = ax.scatter([], [], s=0, marker=">" if b > a else "<", color=DIM_EDGE, zorder=3)
        arrow_heads.append(head)
        txt = ax.text((px[a]+px[b])/2, y0+0.18, "", ha="center", va="bottom", fontsize=8.2, color=DIM_TEXT, zorder=5)
        arrow_labels.append(txt)
    pulse = ax.scatter([], [], s=0, color=PALETTE["blue"], zorder=6, edgecolors="white", linewidths=1.2)
    caption = ax.text((px[0]+px[-1])/2, bottom_y-0.35, "", ha="center", va="center", fontsize=10.5, color="#4B5563", style="italic")
    travel_frames, hold_frames = 14, 10; per_step = travel_frames + hold_frames; total = per_step * len(steps) + 20
    def reset_all():
        for line, head, txt in zip(arrow_lines, arrow_heads, arrow_labels):
            line.set_data([], []); line.set_color(DIM_EDGE); head.set_sizes([0]); txt.set_text(""); txt.set_color(DIM_TEXT)
        pulse.set_sizes([0]); caption.set_text("")
    def draw_arrow_full(k, color):
        a, b, label = steps[k]; y0 = step_y[k]
        arrow_lines[k].set_data([px[a], px[b]], [y0, y0]); arrow_lines[k].set_color(color)
        arrow_heads[k].set_offsets([[px[b], y0]]); arrow_heads[k].set_sizes([120]); arrow_heads[k].set_color(color)
        arrow_labels[k].set_text(label); arrow_labels[k].set_color("#1F2937")
    def update(frame):
        f = frame % (total + 14)
        if f == 0: reset_all()
        if f >= total:
            pulse.set_sizes([0]); caption.set_text("\u2713 Every number comes from this request path.")
            return []
        step_idx = min(f // per_step, len(steps)-1); within = f % per_step
        a, b, label = steps[step_idx]; y0 = step_y[step_idx]
        if within < travel_frames:
            t = ease_in_out(within / travel_frames)
            cx = px[a] + (px[b]-px[a]) * t
            arrow_lines[step_idx].set_data([px[a], cx], [y0, y0]); arrow_lines[step_idx].set_color(PALETTE["blue"])
            pulse.set_offsets([[cx, y0]]); pulse.set_sizes([130])
            arrow_labels[step_idx].set_text(label); arrow_labels[step_idx].set_color("#1F2937")
            caption.set_text(f"Step {step_idx+1}/{len(steps)}: {label}")
        else:
            draw_arrow_full(step_idx, PALETTE["green"]); pulse.set_sizes([0])
            caption.set_text(f"Step {step_idx+1}/{len(steps)}: {label}")
        return []
    anim = FuncAnimation(fig, update, frames=total+30, interval=100, blit=False)
    out_path = f"{OUT_DIR}/02_case_scoring_sequence.gif"
    anim.save(out_path, writer=PillowWriter(fps=10))
    plt.close(fig); print(f"Saved: {out_path}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    animate_row(STAGES_1, CAPTIONS_1, "Prior-Authorization Appeal Outcome Model \u2014 End-to-End System", "01_system_architecture.gif")
    animate_sequence()

if __name__ == "__main__":
    main()
