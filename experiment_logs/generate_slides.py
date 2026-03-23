#!/usr/bin/env python3
"""
Generate presentation slides as a multi-page PDF using matplotlib.
Title: Training Scientific Reasoning in LLMs via Hypothesis-Driven Experimentation
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import numpy as np
import os

# ── Style constants ──────────────────────────────────────────────────────────
TITLE_COLOR = '#1a365d'
SUBTITLE_COLOR = '#2c5282'
BODY_COLOR = '#1a202c'
MUTED_COLOR = '#718096'
ACCENT_BLUE = '#3182ce'
ACCENT_GREEN = '#38a169'
ACCENT_RED = '#e53e3e'
GOOD_BG = '#ebf8ff'
BAD_BG = '#fff5f5'
LIGHT_GRAY = '#f7fafc'
BORDER_GRAY = '#e2e8f0'
WHITE = '#ffffff'
NODE_GREEN = '#c6f6d5'
NODE_RED = '#fed7d7'
NODE_BLUE = '#bee3f8'
NODE_YELLOW = '#fefcbf'
NODE_ORANGE = '#feebc8'

FONT = 'DejaVu Sans'
FIG_W, FIG_H = 16, 9

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presentation.pdf')


def new_slide(fig_w=FIG_W, fig_h=FIG_H):
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)
    return fig, ax


def draw_box(ax, x, y, w, h, text, facecolor=NODE_BLUE, edgecolor='#4a5568',
             fontsize=9, textcolor=BODY_COLOR, bold=False, alpha=1.0, linewidth=1.2,
             ha='center', va='center', fontfamily=FONT, zorder=2, wrap_width=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=linewidth, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha=ha, va=va, fontsize=fontsize,
            color=textcolor, fontfamily=fontfamily, fontweight=weight, zorder=zorder+1,
            wrap=True if wrap_width else False)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color='#4a5568', lw=1.5, style='->', zorder=1):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=zorder)


def slide_title(ax, title, subtitle=None, y_title=8.3, y_sub=7.7):
    ax.text(0.8, y_title, title, fontsize=28, fontweight='bold', color=TITLE_COLOR,
            fontfamily=FONT, va='top')
    if subtitle:
        ax.text(0.8, y_sub, subtitle, fontsize=16, color=SUBTITLE_COLOR,
                fontfamily=FONT, va='top')


def bullet_list(ax, items, x=1.0, y_start=6.8, spacing=0.52, fontsize=14,
                color=BODY_COLOR, bullet='  \u2022  '):
    for i, item in enumerate(items):
        ax.text(x, y_start - i * spacing, f'{bullet}{item}', fontsize=fontsize,
                color=color, fontfamily=FONT, va='top')
    return y_start - len(items) * spacing


def add_footer(ax, slide_num, total=20):
    ax.plot([0.5, 15.5], [0.3, 0.3], color=BORDER_GRAY, linewidth=0.8)
    ax.text(15.5, 0.12, f'{slide_num}/{total}', fontsize=9, color=MUTED_COLOR,
            fontfamily=FONT, ha='right')
    ax.text(0.5, 0.12, 'MLScientist  |  March 2026', fontsize=9, color=MUTED_COLOR,
            fontfamily=FONT)


# ══════════════════════════════════════════════════════════════════════════════
#  SLIDE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def slide_01_title():
    fig, ax = new_slide()
    # Background accent bar
    ax.add_patch(FancyBboxPatch((0, 0), 16, 2.2, boxstyle="square",
                                 facecolor=TITLE_COLOR, edgecolor='none'))
    ax.add_patch(FancyBboxPatch((0, 2.2), 16, 0.08, boxstyle="square",
                                 facecolor=ACCENT_BLUE, edgecolor='none'))

    ax.text(8, 6.2, 'Training Scientific Reasoning in LLMs\nvia Hypothesis-Driven Experimentation',
            fontsize=32, fontweight='bold', color=TITLE_COLOR, fontfamily=FONT,
            ha='center', va='center', linespacing=1.4)

    ax.text(8, 4.4, 'LLM-Guided Tree Search  |  Value of Information Rewards  |  MLGym Benchmark',
            fontsize=15, color=SUBTITLE_COLOR, fontfamily=FONT, ha='center')

    ax.text(8, 1.1, 'March 2026', fontsize=16, color=WHITE, fontfamily=FONT,
            ha='center', fontweight='bold')
    return fig


def slide_02_problem():
    fig, ax = new_slide()
    slide_title(ax, 'Problem: Automating ML Experiment Design')
    add_footer(ax, 2)

    # Left column: text
    items = [
        'A human researcher spends most time deciding what to try next',
        'Read results \u2192 Diagnose \u2192 Decide \u2192 Allocate budget',
        'Goal: Train a "scientist" LLM to make these decisions',
        'Benchmark: MLGym \u2014 9 diverse tasks (classification, regression, RL, game theory)',
    ]
    bullet_list(ax, items, x=0.8, y_start=7.2, fontsize=13, spacing=0.55)

    # Architecture diagram
    y_base = 3.0
    components = [
        (1.5, y_base, 2.5, 1.0, 'Scientist LLM\n(Qwen3-4B / GPT-4o)', '#dbeafe', True),
        (5.2, y_base, 2.5, 1.0, 'Executor LLM\n(Qwen3-4B)', '#e0e7ff', True),
        (8.9, y_base, 2.5, 1.0, 'MLGym Container\n(Docker/Apptainer)', '#fef3c7', True),
        (12.6, y_base, 2.5, 1.0, 'Experiment Tree\n+ Memory', '#d1fae5', True),
    ]
    for (x, y, w, h, text, fc, bold) in components:
        draw_box(ax, x, y, w, h, text, facecolor=fc, fontsize=10, bold=bold)

    # Arrows
    draw_arrow(ax, 4.0, y_base+0.5, 5.2, y_base+0.5, color=ACCENT_BLUE, lw=2)
    ax.text(4.6, y_base+0.75, 'proposes\ndirection', fontsize=8, color=MUTED_COLOR, ha='center', fontfamily=FONT)

    draw_arrow(ax, 7.7, y_base+0.5, 8.9, y_base+0.5, color=ACCENT_BLUE, lw=2)
    ax.text(8.3, y_base+0.75, 'writes\ncode', fontsize=8, color=MUTED_COLOR, ha='center', fontfamily=FONT)

    draw_arrow(ax, 11.4, y_base+0.5, 12.6, y_base+0.5, color=ACCENT_GREEN, lw=2)
    ax.text(12.0, y_base+0.75, 'results\n& scores', fontsize=8, color=MUTED_COLOR, ha='center', fontfamily=FONT)

    # Feedback loop
    draw_arrow(ax, 12.6, y_base-0.1, 4.0, y_base-0.1, color=ACCENT_GREEN, lw=2, style='->')
    ax.text(8, y_base-0.5, 'tree state + memory fed back to Scientist', fontsize=10,
            color=ACCENT_GREEN, ha='center', fontfamily=FONT, fontstyle='italic')

    # Role table
    y_table = 1.3
    ax.add_patch(FancyBboxPatch((1.0, y_table-0.05), 14, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=LIGHT_GRAY, edgecolor=BORDER_GRAY, linewidth=1))
    roles = [
        ('Scientist', 'Reads tree, diagnoses failures, proposes next experiment', 'Qwen3-4B / GPT-4o / Claude'),
        ('Executor', 'Writes and runs code to implement the direction', 'Qwen3-4B'),
        ('Container', 'Isolated env for execution, validation, scoring', 'MLGym Docker/Apptainer'),
    ]
    for i, (role, desc, model) in enumerate(roles):
        y_r = y_table + 0.9 - i * 0.35
        ax.text(1.5, y_r, role, fontsize=10, fontweight='bold', color=TITLE_COLOR, fontfamily=FONT, va='center')
        ax.text(4.0, y_r, desc, fontsize=10, color=BODY_COLOR, fontfamily=FONT, va='center')
        ax.text(12.5, y_r, model, fontsize=9, color=MUTED_COLOR, fontfamily=FONT, va='center')
    return fig


def slide_03_aira_problems():
    fig, ax = new_slide()
    slide_title(ax, 'AIRA / MCTS: The Duhem-Quine Problem')
    add_footer(ax, 3)

    # Draw MCTS tree with UCB scores
    # Root
    draw_box(ax, 3.5, 6.4, 2.8, 0.7, 'Root: Baseline\nscore = 0.764', facecolor=NODE_BLUE, fontsize=10, bold=True)

    # Level 1
    draw_box(ax, 0.5, 4.8, 2.6, 0.7, 'Random Forest\nscore = 0.88   UCB: 1.12', facecolor=NODE_GREEN, fontsize=9)
    draw_box(ax, 4.0, 4.8, 2.6, 0.7, 'SVM\nscore = 0.72   UCB: 0.95', facecolor=NODE_YELLOW, fontsize=9)
    draw_box(ax, 7.5, 4.8, 2.6, 0.7, 'XGBoost\nscore = 0.85   UCB: 1.08', facecolor=NODE_GREEN, fontsize=9)

    # Level 2 under RF
    draw_box(ax, 0.0, 3.0, 2.7, 0.85, 'Add Feature Eng.\nKeyError: "FamilySize"\nScore = 0  (CRASH)', facecolor=NODE_RED, fontsize=8.5)
    draw_box(ax, 3.2, 3.0, 2.5, 0.7, 'Hyperparameter Tune\nscore = 0.89', facecolor=NODE_GREEN, fontsize=9)

    # Arrows
    draw_arrow(ax, 4.9, 6.4, 1.8, 5.5, color='#4a5568', lw=1.5)
    draw_arrow(ax, 4.9, 6.4, 5.3, 5.5, color='#4a5568', lw=1.5)
    draw_arrow(ax, 4.9, 6.4, 8.8, 5.5, color='#4a5568', lw=1.5)
    draw_arrow(ax, 1.8, 4.8, 1.3, 3.85, color='#4a5568', lw=1.5)
    draw_arrow(ax, 1.8, 4.8, 4.4, 3.7, color='#4a5568', lw=1.5)

    # Red X on crashed node
    ax.text(2.85, 3.4, '\u2716', fontsize=22, color=ACCENT_RED, fontweight='bold',
            ha='center', va='center', zorder=10)

    # MCTS interpretation box (bad)
    draw_box(ax, 0.0, 1.6, 5.0, 1.0,
             'MCTS sees: Node scored 0 (crash).\n\u2192 Reduce branch value.\n\u2192 Abandon feature engineering.',
             facecolor=BAD_BG, edgecolor=ACCENT_RED, fontsize=10, linewidth=2)
    ax.text(2.5, 2.75, 'What MCTS concludes:', fontsize=10, fontweight='bold',
            color=ACCENT_RED, ha='center', fontfamily=FONT)

    # Human interpretation box (good)
    draw_box(ax, 5.5, 1.6, 5.0, 1.0,
             'Human sees: KeyError in column naming\n\u2192 Fixable code bug, not a bad idea.\n\u2192 Fix the bug and keep exploring!',
             facecolor=GOOD_BG, edgecolor=ACCENT_GREEN, fontsize=10, linewidth=2)
    ax.text(8.0, 2.75, 'What a scientist concludes:', fontsize=10, fontweight='bold',
            color=ACCENT_GREEN, ha='center', fontfamily=FONT)

    # Annotation
    ax.add_patch(FancyBboxPatch((11.0, 2.8), 4.5, 3.5, boxstyle="round,pad=0.2",
                                 facecolor='#faf5ff', edgecolor='#805ad5', linewidth=2))
    ax.text(13.25, 6.0, 'The Duhem-Quine Problem', fontsize=12, fontweight='bold',
            color='#553c9a', ha='center', fontfamily=FONT)
    ax.text(13.25, 5.5, 'When an experiment fails, you\ncannot tell whether:', fontsize=10,
            color='#553c9a', ha='center', fontfamily=FONT)
    ax.text(11.3, 4.7, '  1. The hypothesis was wrong\n      (bad idea: use linear model\n       on nonlinear data)',
            fontsize=9, color=BODY_COLOR, fontfamily=FONT)
    ax.text(11.3, 3.6, '  2. The implementation was buggy\n      (good idea: feature engineering,\n       but KeyError in code)',
            fontsize=9, color=BODY_COLOR, fontfamily=FONT)
    ax.text(13.25, 3.05, 'MCTS conflates both \u2192 bad search', fontsize=9, fontweight='bold',
            color=ACCENT_RED, ha='center', fontfamily=FONT)

    return fig


def slide_04_more_mcts_problems():
    fig, ax = new_slide()
    slide_title(ax, 'Why MCTS Fails for ML Experiments', 'Three fundamental limitations')
    add_footer(ax, 4)

    # Three columns
    cols = [
        ('Value Estimation\nis Intractable', BAD_BG, ACCENT_RED, 0.5, [
            'Chess: value = win probability',
            '  (well-defined scalar)',
            '',
            'ML experiments: value = ',
            '  "how promising is this direction?"',
            '  (depends on full problem structure,',
            '   executor capability, approach',
            '   interactions)',
            '',
            'No principled way to assign a',
            'scalar to "try feature engineering"',
        ]),
        ('Cannot Reason\nAbout Failures', BAD_BG, ACCENT_RED, 5.5, [
            'UCB sees only scores.',
            'Cannot distinguish:',
            '',
            '\u2716 Bad approach',
            '  (linear model on nonlinear data)',
            '',
            '\u2716 Fixable bug',
            '  (KeyError in feature code)',
            '',
            '\u2716 Near-ceiling',
            '  (RF at 0.90, little headroom)',
            '',
            'Evidence: PyTorch nodes ~70% crash',
            'sklearn nodes ~10% crash',
        ]),
        ('Executor\nIndependence', BAD_BG, ACCENT_RED, 10.5, [
            'In Adaptive MCTS, the executor',
            'ignores proposed strategy ~40%',
            'of the time.',
            '',
            'Does simple config changes that',
            'happen to work. The tree structure',
            'becomes meaningless.',
            '',
            'Formula-based selection is not',
            'actually guiding the search \u2014',
            'the executor independently',
            'finds solutions.',
            '',
            'Tree is decorative, not functional.',
        ]),
    ]

    for title, bg, border, x_start, lines in cols:
        ax.add_patch(FancyBboxPatch((x_start, 0.8), 4.5, 6.5, boxstyle="round,pad=0.15",
                                     facecolor=bg, edgecolor=border, linewidth=1.5))
        ax.text(x_start + 2.25, 6.95, title, fontsize=13, fontweight='bold',
                color=border, ha='center', fontfamily=FONT, linespacing=1.3)
        for i, line in enumerate(lines):
            ax.text(x_start + 0.25, 6.3 - i * 0.38, line, fontsize=9.5,
                    color=BODY_COLOR, fontfamily=FONT)

    return fig


def slide_05_our_approach_tree():
    fig, ax = new_slide()
    slide_title(ax, 'Our Approach: LLM-Guided Tree Search',
                'Realistic example on Titanic survival prediction')
    add_footer(ax, 5)

    # Tree nodes
    nodes = {
        'root': (7, 6.5, 2.5, 0.6, 'root [0.766]\nBaseline logistic regression', NODE_BLUE),
        'r0': (2.5, 5.0, 3.0, 0.6, 'root_0 [0.851]\nRandomForest + FamilySize feature', NODE_GREEN),
        'r1': (10.5, 5.0, 3.0, 0.6, 'root_1 [FAIL]\nNeural net (PyTorch MLP)', NODE_RED),
        'r00': (0.5, 3.3, 3.2, 0.6, 'root_0_0 [0.890]\nCatBoost + target encoding', NODE_GREEN),
        'r01': (4.5, 3.3, 3.0, 0.6, 'root_0_1 [0.861]\nAdd IsAlone interaction', NODE_GREEN),
        'r000': (0.5, 1.7, 3.2, 0.6, 'root_0_0_0 [0.901]\nStacking ensemble (best)', '#a7f3d0'),
    }

    for key, (x, y, w, h, text, fc) in nodes.items():
        draw_box(ax, x, y, w, h, text, facecolor=fc, fontsize=9, bold=(key == 'root'))

    # Edges
    draw_arrow(ax, 8.25, 6.5, 4.0, 5.6, color='#4a5568', lw=1.5)
    draw_arrow(ax, 8.25, 6.5, 12.0, 5.6, color='#4a5568', lw=1.5)
    draw_arrow(ax, 4.0, 5.0, 2.1, 3.9, color='#4a5568', lw=1.5)
    draw_arrow(ax, 4.0, 5.0, 6.0, 3.9, color='#4a5568', lw=1.5)
    draw_arrow(ax, 2.1, 3.3, 2.1, 2.3, color='#4a5568', lw=1.5)

    # Red X on failed node
    ax.text(13.7, 5.3, '\u2716', fontsize=20, color=ACCENT_RED, fontweight='bold', zorder=10)

    # Memory annotations (right side)
    ax.add_patch(FancyBboxPatch((9.5, 1.2, ), 6.0, 2.4, boxstyle="round,pad=0.15",
                                 facecolor=GOOD_BG, edgecolor=ACCENT_BLUE, linewidth=1.5))
    ax.text(12.5, 3.35, 'Scientist Memory (learned)', fontsize=11, fontweight='bold',
            color=ACCENT_BLUE, ha='center', fontfamily=FONT)

    memories = [
        '\u2713 "FamilySize = SibSp+Parch+1 improves tree models"',
        '\u2713 "CatBoost handles categoricals natively \u2014 no one-hot"',
        '\u2713 "Target encoding works for high-cardinality (Cabin)"',
        '\u2716 "Neural nets fail on small tabular data (n<1000)"',
        '\u2713 "Stacking RF + CatBoost + XGB yields 0.901"',
    ]
    for i, mem in enumerate(memories):
        c = ACCENT_RED if '\u2716' in mem else ACCENT_GREEN
        ax.text(9.8, 3.0 - i * 0.38, mem, fontsize=8.5, color=c, fontfamily=FONT)

    return fig


def slide_06_our_approach_flow():
    fig, ax = new_slide()
    slide_title(ax, 'LLM-Guided Search: Decision Flow',
                'How the scientist reasons at each step')
    add_footer(ax, 6)

    # Flow diagram (vertical)
    steps = [
        (8, 6.7, 5.0, 0.55, 'Step 1: Read full tree + memory', '#dbeafe'),
        (8, 5.7, 5.0, 0.55, 'Step 2: Inspect action logs of selected nodes', '#e0e7ff'),
        (8, 4.7, 5.0, 0.55, 'Step 3: Diagnose — bug vs bad approach vs ceiling', '#fef3c7'),
        (8, 3.7, 5.0, 0.55, 'Step 4: Brainstorm 3 diverse strategies (P < 0.2 each)', '#d1fae5'),
        (8, 2.7, 5.0, 0.55, 'Step 5: Choose best strategy for remaining budget', '#c6f6d5'),
        (8, 1.7, 5.0, 0.55, 'Step 6: Executor implements → results → update tree', '#bee3f8'),
    ]

    for (cx, y, w, h, text, fc) in steps:
        draw_box(ax, cx - w/2, y, w, h, text, facecolor=fc, fontsize=12, bold=True)

    for i in range(len(steps)-1):
        draw_arrow(ax, 8, steps[i][1], 8, steps[i+1][1]+0.55, color=ACCENT_BLUE, lw=2)

    # Feedback arrow
    ax.annotate('', xy=(13.5, 6.95), xytext=(13.5, 1.95),
                arrowprops=dict(arrowstyle='->', color=ACCENT_GREEN, lw=2,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(14.5, 4.4, 'Loop until\nbudget\nexhausted', fontsize=10, color=ACCENT_GREEN,
            ha='center', fontfamily=FONT, fontstyle='italic')

    # Advantages on left
    ax.add_patch(FancyBboxPatch((0.3, 1.5), 4.5, 5.8, boxstyle="round,pad=0.15",
                                 facecolor=GOOD_BG, edgecolor=ACCENT_BLUE, linewidth=1.5))
    ax.text(2.55, 7.0, 'Key Advantages over MCTS', fontsize=12, fontweight='bold',
            color=ACCENT_BLUE, ha='center', fontfamily=FONT)
    advantages = [
        '1. Reasons about WHY\n   experiments fail (not just\n   that they failed)',
        '2. Builds on success:\n   "log transform helped GrLivArea\n   \u2192 try other skewed features"',
        '3. Counterfactual reasoning:\n   "target encoding instead of\n   one-hot avoids cardinality"',
        '4. Budget-aware:\n   "5+ nodes left \u2192 explore.\n   2 or fewer \u2192 refine best."',
    ]
    for i, adv in enumerate(advantages):
        ax.text(0.6, 6.5 - i * 1.3, adv, fontsize=9.5, color=BODY_COLOR, fontfamily=FONT,
                linespacing=1.3)

    return fig


def slide_07_battle_of_sexes():
    fig, ax = new_slide()
    slide_title(ax, 'Qualitative Success: Battle of Sexes',
                'SFT scientist (1.44) vs baseline (1.24)')
    add_footer(ax, 7)

    # Reasoning chain as connected boxes
    chain = [
        ('Observation', 'Opponent copies my last\nmove with ~80% probability', '#dbeafe', 0),
        ('Attempt 1', 'Mirror strategies\nScore: 0.54 \u2014 predictability\nis exploited by opponent', NODE_RED, 1),
        ('Attempt 2', 'Periodic pattern (AABBA...)\nScore: 1.30 \u2014 breaks copying\nbut still partially predictable', NODE_YELLOW, 2),
        ('Attempt 3', 'Adaptive reinforcement-based\nfeedback loop\nScore: 1.44 \u2014 best exploit!', NODE_GREEN, 3),
    ]

    x_start = 1.0
    box_w = 3.2
    box_h = 1.5
    gap = 0.5

    for i, (label, text, fc, idx) in enumerate(chain):
        x = x_start + i * (box_w + gap)
        draw_box(ax, x, 3.5, box_w, box_h, text, facecolor=fc, fontsize=10)
        ax.text(x + box_w/2, 5.2, label, fontsize=11, fontweight='bold',
                color=TITLE_COLOR, ha='center', fontfamily=FONT)
        if i < len(chain) - 1:
            draw_arrow(ax, x + box_w, 4.25, x + box_w + gap, 4.25,
                      color=ACCENT_BLUE, lw=2.5)

    # Score comparison bar
    ax.add_patch(FancyBboxPatch((1.0, 1.0), 14, 1.8, boxstyle="round,pad=0.15",
                                 facecolor=LIGHT_GRAY, edgecolor=BORDER_GRAY, linewidth=1))

    bar_y = 1.8
    bar_h = 0.5
    # Baseline
    baseline_w = 1.24 * 3.5
    ax.add_patch(FancyBboxPatch((2.0, bar_y + 0.55), baseline_w, bar_h, boxstyle="round,pad=0.05",
                                 facecolor='#cbd5e0', edgecolor='#4a5568'))
    ax.text(2.0 + baseline_w + 0.2, bar_y + 0.8, 'Baseline: 1.24', fontsize=11, color=MUTED_COLOR,
            fontfamily=FONT, va='center')

    # SFT
    sft_w = 1.44 * 3.5
    ax.add_patch(FancyBboxPatch((2.0, bar_y - 0.05), sft_w, bar_h, boxstyle="round,pad=0.05",
                                 facecolor=ACCENT_BLUE, edgecolor='#2b6cb0'))
    ax.text(2.0 + sft_w + 0.2, bar_y + 0.2, 'SFT Scientist: 1.44  (+16%)', fontsize=11,
            color=ACCENT_BLUE, fontfamily=FONT, va='center', fontweight='bold')

    ax.text(8, 6.3, 'Scientist correctly identified opponent copying behavior and designed an adaptive exploitation strategy',
            fontsize=12, color=BODY_COLOR, ha='center', fontfamily=FONT, fontstyle='italic')

    return fig


def slide_08_hallucination_failure():
    fig, ax = new_slide()
    slide_title(ax, 'Qualitative Failure: Domain Hallucination',
                'SFT scientist applies Titanic reasoning to game theory')
    add_footer(ax, 8)

    # Left: Task description
    ax.add_patch(FancyBboxPatch((0.5, 1.5), 6.5, 5.5, boxstyle="round,pad=0.15",
                                 facecolor=GOOD_BG, edgecolor=ACCENT_BLUE, linewidth=2))
    ax.text(3.75, 6.7, "Prisoner's Dilemma — Actual Task", fontsize=13, fontweight='bold',
            color=ACCENT_BLUE, ha='center', fontfamily=FONT)
    task_lines = [
        'Payoff matrix:',
        '            Cooperate    Defect',
        ' Cooperate    (3, 3)       (0, 5)',
        ' Defect       (5, 0)       (1, 1)',
        '',
        'Goal: maximize total payoff over',
        '200 rounds against adaptive opponent.',
        '',
        'Key concepts:',
        '  \u2022 Tit-for-tat, forgiveness',
        '  \u2022 Exploitation vs cooperation',
        '  \u2022 Nash equilibrium (defect/defect)',
    ]
    for i, line in enumerate(task_lines):
        ax.text(0.8, 6.2 - i * 0.38, line, fontsize=10, color=BODY_COLOR,
                fontfamily='DejaVu Sans Mono' if i in [1,2,3] else FONT)

    # Right: Scientist's actual output
    ax.add_patch(FancyBboxPatch((8.0, 1.5), 7.5, 5.5, boxstyle="round,pad=0.15",
                                 facecolor=BAD_BG, edgecolor=ACCENT_RED, linewidth=2))
    ax.text(11.75, 6.7, 'SFT Scientist Output (WRONG)', fontsize=13, fontweight='bold',
            color=ACCENT_RED, ha='center', fontfamily=FONT)
    bad_lines = [
        '"We should analyze the survival rate',
        ' by Pclass and create interaction',
        ' features between Sex and Embarked..."',
        '',
        '"FamilySize = SibSp + Parch + 1',
        ' should improve the Random Forest..."',
        '',
        '"Consider log transform on Fare',
        ' and impute Age with median..."',
    ]
    for i, line in enumerate(bad_lines):
        ax.text(8.3, 6.2 - i * 0.38, line, fontsize=10, color=ACCENT_RED,
                fontfamily=FONT, fontstyle='italic')

    # Big RED highlight
    ax.text(11.75, 2.4, 'COMPLETE DOMAIN MISMATCH', fontsize=14, fontweight='bold',
            color=ACCENT_RED, ha='center', fontfamily=FONT)
    ax.text(11.75, 1.9, 'Model memorized Titanic patterns\ninstead of learning general reasoning',
            fontsize=11, color=ACCENT_RED, ha='center', fontfamily=FONT)

    # Arrow between
    ax.annotate('', xy=(8.0, 4.25), xytext=(7.0, 4.25),
                arrowprops=dict(arrowstyle='->', color=ACCENT_RED, lw=3))
    ax.text(7.5, 4.75, 'Model\nignores\ntask!', fontsize=10, fontweight='bold',
            color=ACCENT_RED, ha='center', fontfamily=FONT, rotation=0)

    return fig


def slide_09_why_ttt():
    fig, ax = new_slide()
    slide_title(ax, 'Why Test-Time Training (SFT / RL)?',
                'Distilling search-time knowledge into model weights')
    add_footer(ax, 9)

    # Problems with search only
    ax.add_patch(FancyBboxPatch((0.5, 3.8), 7.0, 3.5, boxstyle="round,pad=0.15",
                                 facecolor=BAD_BG, edgecolor=ACCENT_RED, linewidth=1.5))
    ax.text(4.0, 7.0, 'Problems with Search Only', fontsize=14, fontweight='bold',
            color=ACCENT_RED, ha='center', fontfamily=FONT)
    problems = [
        '\u2022 Context window limits: tree > 15 nodes overflows context',
        '\u2022 Compute cost: each scientist call = full LLM forward pass',
        '\u2022 Executor calls: multiple tool-use turns per node',
        '\u2022 Diminishing returns: later nodes are refinements, not breakthroughs',
        '\u2022 No learning: same mistakes repeated across different runs',
    ]
    for i, p in enumerate(problems):
        ax.text(0.8, 6.5 - i * 0.52, p, fontsize=11, color=BODY_COLOR, fontfamily=FONT)

    # Goal
    ax.add_patch(FancyBboxPatch((8.0, 3.8), 7.5, 3.5, boxstyle="round,pad=0.15",
                                 facecolor=GOOD_BG, edgecolor=ACCENT_GREEN, linewidth=1.5))
    ax.text(11.75, 7.0, 'Goal of Test-Time Training', fontsize=14, fontweight='bold',
            color=ACCENT_GREEN, ha='center', fontfamily=FONT)
    goals = [
        '\u2022 Distill search knowledge into weights',
        '\u2022 Propose better experiments earlier',
        '\u2022 Need fewer nodes to find good solutions',
        '\u2022 Internalize patterns from prior trajectories',
        '\u2022 Transfer insights across similar tasks',
    ]
    for i, g in enumerate(goals):
        ax.text(8.3, 6.5 - i * 0.52, g, fontsize=11, color=BODY_COLOR, fontfamily=FONT)

    # Pipeline diagram at bottom
    pipeline_steps = [
        (1.5, 1.5, 3.0, 1.2, 'Tree Search\nTrajectories\n(run experiments)', '#dbeafe'),
        (5.5, 1.5, 3.0, 1.2, 'Extract Scientist\nDecisions + Outcomes\n(training data)', '#e0e7ff'),
        (9.5, 1.5, 3.0, 1.2, 'SFT / GRPO\nTraining\n(update weights)', '#fef3c7'),
        (13.0, 1.5, 2.5, 1.2, 'Better Scientist\n(fewer nodes,\nhigher scores)', '#d1fae5'),
    ]
    for (x, y, w, h, text, fc) in pipeline_steps:
        draw_box(ax, x, y, w, h, text, facecolor=fc, fontsize=10, bold=True)
    for i in range(len(pipeline_steps) - 1):
        x1 = pipeline_steps[i][0] + pipeline_steps[i][2]
        x2 = pipeline_steps[i+1][0]
        draw_arrow(ax, x1, 2.1, x2, 2.1, color=ACCENT_BLUE, lw=2.5)

    return fig


def slide_10_sft_pipeline():
    fig, ax = new_slide()
    slide_title(ax, 'SFT Pipeline: From Search to Supervision')
    add_footer(ax, 10)

    # Three-stage diagram
    # Stage 1: Data Collection
    ax.add_patch(FancyBboxPatch((0.5, 4.0), 4.5, 3.0, boxstyle="round,pad=0.15",
                                 facecolor='#dbeafe', edgecolor=ACCENT_BLUE, linewidth=1.5))
    ax.text(2.75, 6.8, 'Data Collection', fontsize=13, fontweight='bold',
            color=ACCENT_BLUE, ha='center', fontfamily=FONT)
    dc_items = [
        'Run tree search on MLGym tasks',
        'Collect scientist decisions at each node:',
        '  - Tree state (input)',
        '  - Strategy choice (output)',
        '  - Result & diagnosis (feedback)',
        'Filter: keep successful trajectories',
    ]
    for i, item in enumerate(dc_items):
        ax.text(0.8, 6.3 - i * 0.36, item, fontsize=9.5, color=BODY_COLOR, fontfamily=FONT)

    # Stage 2: Data Formatting
    ax.add_patch(FancyBboxPatch((5.5, 4.0), 4.5, 3.0, boxstyle="round,pad=0.15",
                                 facecolor='#fef3c7', edgecolor='#d69e2e', linewidth=1.5))
    ax.text(7.75, 6.8, 'Data Formatting', fontsize=13, fontweight='bold',
            color='#b7791f', ha='center', fontfamily=FONT)
    df_items = [
        'Two formats tested:',
        '',
        'Focused QA (small_ques):',
        '  Short Q&A grounded in task context',
        '',
        'Deep Think (deep_think):',
        '  Extended reasoning chains with',
        '  explicit task property references',
    ]
    for i, item in enumerate(df_items):
        ax.text(5.8, 6.3 - i * 0.36, item, fontsize=9.5, color=BODY_COLOR, fontfamily=FONT)

    # Stage 3: Training
    ax.add_patch(FancyBboxPatch((10.5, 4.0), 5.0, 3.0, boxstyle="round,pad=0.15",
                                 facecolor='#d1fae5', edgecolor=ACCENT_GREEN, linewidth=1.5))
    ax.text(13.0, 6.8, 'SFT Training', fontsize=13, fontweight='bold',
            color=ACCENT_GREEN, ha='center', fontfamily=FONT)
    st_items = [
        'Model: Qwen3-4B',
        'Per-task training (separate models)',
        'Early stopping (epoch 1-3 best)',
        'LoRA fine-tuning (efficient)',
        '',
        'Key finding: per-task >> multi-task',
        '  Per-task avg delta: +0.016',
        '  Multi-task avg delta: -0.018',
    ]
    for i, item in enumerate(st_items):
        ax.text(10.8, 6.3 - i * 0.36, item, fontsize=9.5, color=BODY_COLOR, fontfamily=FONT)

    # Arrows between stages
    draw_arrow(ax, 5.0, 5.5, 5.5, 5.5, color=ACCENT_BLUE, lw=2.5)
    draw_arrow(ax, 10.0, 5.5, 10.5, 5.5, color='#d69e2e', lw=2.5)

    # Bottom: data evolution
    ax.add_patch(FancyBboxPatch((0.5, 0.8), 15.0, 2.5, boxstyle="round,pad=0.15",
                                 facecolor=LIGHT_GRAY, edgecolor=BORDER_GRAY, linewidth=1))
    ax.text(8, 3.05, 'Training Data Evolution', fontsize=13, fontweight='bold',
            color=TITLE_COLOR, ha='center', fontfamily=FONT)

    versions = [
        ('v1: Template QA', '8,170 samples', 'Shallow, no downstream gain', BAD_BG, ACCENT_RED),
        ('v2: Claude-generated', '704 samples', 'Rich but mode collapse / hallucination', NODE_YELLOW, '#b7791f'),
        ('v3: Self-generated\n      grounded', '2,378 QA\n+ 788 traces', 'Task-aware, first significant results!', GOOD_BG, ACCENT_GREEN),
    ]
    for i, (name, count, result, bg, color) in enumerate(versions):
        x = 1.0 + i * 5.0
        ax.add_patch(FancyBboxPatch((x, 1.0), 4.5, 1.7, boxstyle="round,pad=0.1",
                                     facecolor=bg, edgecolor=color, linewidth=1.5))
        ax.text(x + 2.25, 2.4, name, fontsize=10, fontweight='bold', color=color,
                ha='center', fontfamily=FONT, linespacing=1.2)
        ax.text(x + 2.25, 1.85, count, fontsize=9, color=MUTED_COLOR, ha='center', fontfamily=FONT)
        ax.text(x + 2.25, 1.3, result, fontsize=8.5, color=BODY_COLOR, ha='center', fontfamily=FONT)
        if i < len(versions) - 1:
            draw_arrow(ax, x + 4.5, 1.85, x + 5.0, 1.85, color=MUTED_COLOR, lw=2)

    return fig


def slide_11_training_data():
    fig, ax = new_slide()
    slide_title(ax, 'Training Data: From Templates to Grounded Traces')
    add_footer(ax, 11)

    # v1
    y = 6.5
    ax.add_patch(FancyBboxPatch((0.5, y-0.1), 15.0, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=BAD_BG, edgecolor=ACCENT_RED, linewidth=1.5))
    ax.text(1.0, y+0.8, 'v1: Template-Based QA (8,170 samples)', fontsize=13, fontweight='bold',
            color=ACCENT_RED, fontfamily=FONT)
    ax.text(1.0, y+0.25, 'Q: "What would happen if we used XGBoost instead of RF?"    A: [template-derived answer]',
            fontsize=10, color=BODY_COLOR, fontfamily='DejaVu Sans Mono')
    ax.text(10.0, y+0.25, 'Loss: 2.16 \u2192 0.07, Perplexity: 36 \u2192 1.1', fontsize=10,
            color=MUTED_COLOR, fontfamily=FONT)
    ax.text(14.0, y+0.8, 'Zero downstream gain', fontsize=10, fontweight='bold',
            color=ACCENT_RED, fontfamily=FONT)

    # v2
    y = 4.5
    ax.add_patch(FancyBboxPatch((0.5, y-0.1), 15.0, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=NODE_YELLOW, edgecolor='#b7791f', linewidth=1.5))
    ax.text(1.0, y+0.8, 'v2: Claude-Generated Reasoning (704 samples, ~$4.37)', fontsize=13, fontweight='bold',
            color='#b7791f', fontfamily=FONT)
    ax.text(1.0, y+0.25, 'Rich reasoning with counterfactual analysis and literature references',
            fontsize=10, color=BODY_COLOR, fontfamily=FONT)
    ax.text(10.0, y+0.25, 'R^2 +0.009 (p=0.10) on house price only', fontsize=10,
            color=MUTED_COLOR, fontfamily=FONT)
    ax.text(14.0, y+0.8, 'Hallucination problem', fontsize=10, fontweight='bold',
            color='#b7791f', fontfamily=FONT)

    # v3
    y = 2.5
    ax.add_patch(FancyBboxPatch((0.5, y-0.1), 15.0, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=GOOD_BG, edgecolor=ACCENT_GREEN, linewidth=1.5))
    ax.text(1.0, y+0.8, 'v3: Grounded Self-Generated (2,378 QA + 788 traces)', fontsize=13, fontweight='bold',
            color=ACCENT_GREEN, fontfamily=FONT)
    ax.text(1.0, y+0.25, 'Task YAML description in every sample. Model conditions reasoning on actual task.',
            fontsize=10, color=BODY_COLOR, fontfamily=FONT)
    ax.text(10.0, y+0.25, 'Titanic +0.046 (p=0.032), +0.031 (p=0.003)', fontsize=10,
            color=ACCENT_GREEN, fontfamily=FONT, fontweight='bold')
    ax.text(14.0, y+0.8, 'First significant results!', fontsize=10, fontweight='bold',
            color=ACCENT_GREEN, fontfamily=FONT)

    # Key takeaway
    ax.add_patch(FancyBboxPatch((2.0, 0.5), 12.0, 0.9, boxstyle="round,pad=0.1",
                                 facecolor='#faf5ff', edgecolor='#805ad5', linewidth=2))
    ax.text(8.0, 0.95, 'Lesson: Offline metrics (loss, perplexity) do NOT predict downstream performance.',
            fontsize=12, fontweight='bold', color='#553c9a', ha='center', fontfamily=FONT)

    return fig


def slide_12_grounding_fix():
    fig, ax = new_slide()
    slide_title(ax, 'The Grounding Fix: Task Descriptions Matter',
                'The difference between memorization and reasoning')
    add_footer(ax, 12)

    # Before (left)
    ax.add_patch(FancyBboxPatch((0.5, 1.2), 7.0, 5.8, boxstyle="round,pad=0.15",
                                 facecolor=BAD_BG, edgecolor=ACCENT_RED, linewidth=2))
    ax.text(4.0, 6.7, 'BEFORE: No Task Description (v2)', fontsize=13, fontweight='bold',
            color=ACCENT_RED, ha='center', fontfamily=FONT)

    ax.text(0.8, 6.1, 'Training sample:', fontsize=10, fontweight='bold', color=BODY_COLOR, fontfamily=FONT)
    ax.text(0.8, 5.6, '  input:  [tree state only]', fontsize=9.5, color=MUTED_COLOR, fontfamily='DejaVu Sans Mono')
    ax.text(0.8, 5.2, '  output: [Claude reasoning trace]', fontsize=9.5, color=MUTED_COLOR, fontfamily='DejaVu Sans Mono')

    ax.text(0.8, 4.5, 'Result on Prisoner\'s Dilemma:', fontsize=10, fontweight='bold', color=BODY_COLOR, fontfamily=FONT)
    bad_output = [
        '"We should analyze the survival rate',
        ' by Pclass and create interaction',
        ' features between Sex and Embarked.',
        ' FamilySize = SibSp + Parch + 1',
        ' should improve the Random Forest."',
    ]
    for i, line in enumerate(bad_output):
        ax.text(0.8, 4.0 - i * 0.35, line, fontsize=9.5, color=ACCENT_RED, fontfamily=FONT, fontstyle='italic')

    ax.text(4.0, 1.7, 'Titanic patterns on game theory!', fontsize=11, fontweight='bold',
            color=ACCENT_RED, ha='center', fontfamily=FONT)

    # After (right)
    ax.add_patch(FancyBboxPatch((8.0, 1.2), 7.5, 5.8, boxstyle="round,pad=0.15",
                                 facecolor=GOOD_BG, edgecolor=ACCENT_GREEN, linewidth=2))
    ax.text(11.75, 6.7, 'AFTER: With Task Description (v3)', fontsize=13, fontweight='bold',
            color=ACCENT_GREEN, ha='center', fontfamily=FONT)

    ax.text(8.3, 6.1, 'Training sample:', fontsize=10, fontweight='bold', color=BODY_COLOR, fontfamily=FONT)
    ax.text(8.3, 5.6, '  input:  [task YAML + tree state]', fontsize=9.5, color=ACCENT_GREEN, fontfamily='DejaVu Sans Mono')
    ax.text(8.3, 5.2, '  output: [self-generated reasoning]', fontsize=9.5, color=ACCENT_GREEN, fontfamily='DejaVu Sans Mono')

    ax.text(8.3, 4.5, 'Result on Battle of Sexes:', fontsize=10, fontweight='bold', color=BODY_COLOR, fontfamily=FONT)
    good_output = [
        '"This is a non-zero-sum coordination',
        ' game with asymmetric preferences.',
        ' Opponent copies with ~80% probability.',
        ' Adaptive reinforcement-based feedback',
        ' should exploit the copying behavior."',
    ]
    for i, line in enumerate(good_output):
        ax.text(8.3, 4.0 - i * 0.35, line, fontsize=9.5, color=ACCENT_GREEN, fontfamily=FONT, fontstyle='italic')

    ax.text(11.75, 1.7, 'Correct domain reasoning!', fontsize=11, fontweight='bold',
            color=ACCENT_GREEN, ha='center', fontfamily=FONT)

    # Arrow
    ax.annotate('', xy=(8.0, 4.0), xytext=(7.5, 4.0),
                arrowprops=dict(arrowstyle='->', color=TITLE_COLOR, lw=3))

    return fig


def slide_13_sft_results():
    fig, ax = new_slide()
    slide_title(ax, 'SFT Results: Full Evaluation (n=5)')
    add_footer(ax, 13)

    # Table
    headers = ['Model', 'Titanic', 'Regression', 'BoS', 'Prisoner', 'MountainCar']
    rows = [
        ['baseline', '0.827', '0.882', '1.02', '2.37', '33.8'],
        ['sq_e1', '0.873*', '0.885', '1.27', '2.35', '35.1'],
        ['sq_e2', '0.849', '0.884', '1.24', '2.36', '34.2'],
        ['sq_e3', '0.841', '0.883', '1.30', '2.33', '33.9'],
        ['dt_e1', '0.845', '0.881', '0.80', '2.34', '34.0'],
        ['dt_titani_e6', '0.858*', '0.880', '1.18', '2.35', '34.5'],
        ['multi_task', '0.809', '0.879', '0.99', '2.31', '33.2'],
    ]

    # Draw table
    col_x = [1.0, 3.5, 5.8, 8.0, 10.0, 12.2]
    col_w = [2.0, 2.0, 1.8, 1.8, 1.8, 2.0]

    # Header
    for j, h in enumerate(headers):
        ax.add_patch(FancyBboxPatch((col_x[j], 6.2), col_w[j], 0.55, boxstyle="square",
                                     facecolor=TITLE_COLOR, edgecolor=WHITE, linewidth=0.5))
        ax.text(col_x[j] + col_w[j]/2, 6.47, h, fontsize=11, fontweight='bold',
                color=WHITE, ha='center', va='center', fontfamily=FONT)

    # Rows
    for i, row in enumerate(rows):
        y = 5.6 - i * 0.55
        bg = LIGHT_GRAY if i % 2 == 0 else WHITE
        for j, cell in enumerate(row):
            cell_bg = bg
            cell_color = BODY_COLOR
            cell_weight = 'normal'
            if '*' in cell:
                cell_bg = '#d1fae5'
                cell_color = ACCENT_GREEN
                cell_weight = 'bold'
            elif j == 0 and 'multi' in cell:
                cell_bg = BAD_BG
                cell_color = ACCENT_RED
            ax.add_patch(FancyBboxPatch((col_x[j], y), col_w[j], 0.55, boxstyle="square",
                                         facecolor=cell_bg, edgecolor=BORDER_GRAY, linewidth=0.5))
            ax.text(col_x[j] + col_w[j]/2, y + 0.27, cell.replace('*', ''),
                    fontsize=10, color=cell_color, fontweight=cell_weight,
                    ha='center', va='center', fontfamily=FONT)
            if '*' in cell:
                ax.text(col_x[j] + col_w[j] - 0.15, y + 0.45, '*', fontsize=8,
                        color=ACCENT_GREEN, fontweight='bold', fontfamily=FONT)

    # Legend
    ax.text(1.0, 1.7, '* = statistically significant (p < 0.05)', fontsize=11, fontweight='bold',
            color=ACCENT_GREEN, fontfamily=FONT)
    ax.text(1.0, 1.2, 'sq = small_ques format | dt = deep_think format | e1-6 = epoch number',
            fontsize=10, color=MUTED_COLOR, fontfamily=FONT)

    # Key findings box
    ax.add_patch(FancyBboxPatch((8.0, 0.7), 7.5, 1.3, boxstyle="round,pad=0.1",
                                 facecolor=GOOD_BG, edgecolor=ACCENT_BLUE, linewidth=1.5))
    ax.text(11.75, 1.7, 'Key Findings', fontsize=11, fontweight='bold', color=ACCENT_BLUE,
            ha='center', fontfamily=FONT)
    ax.text(8.3, 1.3, '\u2022 Per-task training works: sq_e1 Titanic +0.046 (p=0.032)', fontsize=9.5,
            color=BODY_COLOR, fontfamily=FONT)
    ax.text(8.3, 0.95, '\u2022 Multi-task consistently hurts across all tasks', fontsize=9.5,
            color=BODY_COLOR, fontfamily=FONT)

    return fig


def slide_14_in_vs_out_domain():
    fig, ax = new_slide()
    slide_title(ax, 'In-Domain vs Out-of-Domain Performance',
                'Per-task training helps in-domain but does not transfer')
    add_footer(ax, 14)

    # Bar chart
    categories = ['Titanic\n(in-domain)', 'Regression\n(in-domain)', 'BoS\n(out-domain)', 'Prisoner\n(out-domain)', 'MountainCar\n(out-domain)']
    baseline = [0.827, 0.882, 1.02, 2.37, 0.338]  # normalize MC
    best_sft = [0.873, 0.885, 1.30, 2.35, 0.351]  # normalize MC
    deltas = [b - a for a, b in zip(baseline, best_sft)]

    x = np.arange(len(categories))
    width = 0.35

    # Normalize for display (bar chart of deltas)
    ax_chart = fig.add_axes([0.08, 0.15, 0.85, 0.55])
    ax_chart.set_facecolor(WHITE)

    colors_bars = [ACCENT_GREEN, ACCENT_GREEN, ACCENT_BLUE, ACCENT_RED, ACCENT_BLUE]
    bars = ax_chart.bar(x, deltas, width=0.6, color=colors_bars, edgecolor='white', linewidth=1.5)

    ax_chart.set_xticks(x)
    ax_chart.set_xticklabels(categories, fontsize=11, fontfamily=FONT)
    ax_chart.set_ylabel('Score Delta (best SFT - baseline)', fontsize=12, fontfamily=FONT)
    ax_chart.axhline(y=0, color='#4a5568', linewidth=1)
    ax_chart.spines['top'].set_visible(False)
    ax_chart.spines['right'].set_visible(False)

    for bar, d in zip(bars, deltas):
        y_pos = bar.get_height() + 0.002 if d >= 0 else bar.get_height() - 0.008
        ax_chart.text(bar.get_x() + bar.get_width()/2, y_pos, f'{d:+.3f}',
                      ha='center', fontsize=10, fontweight='bold', fontfamily=FONT,
                      color=ACCENT_GREEN if d > 0 else ACCENT_RED)

    # Legend
    ax.add_patch(FancyBboxPatch((1.0, 6.7), 3.5, 0.5, boxstyle="round,pad=0.05",
                                 facecolor=NODE_GREEN, edgecolor=ACCENT_GREEN, linewidth=1))
    ax.text(2.75, 6.95, 'In-domain (trained on)', fontsize=10, ha='center', fontfamily=FONT,
            color=ACCENT_GREEN, fontweight='bold')

    ax.add_patch(FancyBboxPatch((5.0, 6.7), 3.5, 0.5, boxstyle="round,pad=0.05",
                                 facecolor=NODE_BLUE, edgecolor=ACCENT_BLUE, linewidth=1))
    ax.text(6.75, 6.95, 'Out-of-domain (not trained)', fontsize=10, ha='center', fontfamily=FONT,
            color=ACCENT_BLUE, fontweight='bold')

    return fig


def slide_15_n20():
    fig, ax = new_slide()
    slide_title(ax, 'N20 Scaling: More Budget Helps Everyone',
                'Performance scales strongly with search budget (n5 \u2192 n20)')
    add_footer(ax, 15)

    # Table
    headers = ['Task', 'Method', 'n5', 'n20', 'Delta']
    rows = [
        ['Titanic', 'baseline', '0.827', '0.945', '+0.118'],
        ['Regression', 'sq_e1 (SFT)', '0.885', '0.909', '+0.024'],
        ['Regression', 'baseline', '0.882', '0.888', '+0.006'],
        ['BoS', 'sq_battle_e3', '1.371', '~1.433', '+0.062'],
        ['BoS', 'baseline', '--', '1.441', '--'],
    ]

    col_x = [1.5, 4.5, 7.0, 9.0, 11.5]
    col_w = [2.5, 2.2, 1.8, 1.8, 2.0]

    for j, h in enumerate(headers):
        ax.add_patch(FancyBboxPatch((col_x[j], 6.0), col_w[j], 0.55, boxstyle="square",
                                     facecolor=TITLE_COLOR, edgecolor=WHITE, linewidth=0.5))
        ax.text(col_x[j] + col_w[j]/2, 6.27, h, fontsize=12, fontweight='bold',
                color=WHITE, ha='center', va='center', fontfamily=FONT)

    for i, row in enumerate(rows):
        y = 5.35 - i * 0.55
        bg = LIGHT_GRAY if i % 2 == 0 else WHITE
        for j, cell in enumerate(row):
            cell_bg = bg
            if j == 4 and cell.startswith('+0.1'):
                cell_bg = '#d1fae5'
            ax.add_patch(FancyBboxPatch((col_x[j], y), col_w[j], 0.55, boxstyle="square",
                                         facecolor=cell_bg, edgecolor=BORDER_GRAY, linewidth=0.5))
            weight = 'bold' if (j == 4 and cell.startswith('+0.1')) else 'normal'
            ax.text(col_x[j] + col_w[j]/2, y + 0.27, cell,
                    fontsize=11, color=BODY_COLOR, fontweight=weight,
                    ha='center', va='center', fontfamily=FONT)

    # Key findings
    findings = [
        '\u2022 Titanic baseline jumps from 0.827 to 0.945 (+0.118) \u2014 massive gain from budget alone',
        '\u2022 SFT advantage is most pronounced at LOW budget (n5): model compensates for limited search',
        '\u2022 At high budget (n20), baseline catches up \u2014 brute-force search becomes effective',
        '\u2022 Regression near-ceiling: both SFT and baseline converge near 0.885-0.909',
        '\u2022 Open question: does SFT advantage widen or narrow with more budget?',
    ]
    for i, f in enumerate(findings):
        ax.text(1.0, 3.0 - i * 0.48, f, fontsize=11, color=BODY_COLOR, fontfamily=FONT)

    return fig


def slide_16_ceiling():
    fig, ax = new_slide()
    slide_title(ax, 'Task Ceiling Analysis', 'Where to focus next')
    add_footer(ax, 16)

    # Horizontal bar chart
    tasks = ['Mountain Car', 'Battle of Sexes', 'Titanic', 'Prisoner Dilemma', 'Regression']
    baselines = [33.8, 1.02, 0.77, 2.37, 0.88]
    our_best = [68.9, 1.44, 0.945, 2.39, 0.92]
    ceilings = [90.0, 1.55, 0.90, 2.50, 0.93]

    # Normalize everything to 0-1 range for each task
    for i in range(len(tasks)):
        mn = min(baselines[i], 0)
        mx = ceilings[i] * 1.1
        baselines[i] = (baselines[i] - mn) / (mx - mn)
        our_best[i] = (our_best[i] - mn) / (mx - mn)
        ceilings[i] = (ceilings[i] - mn) / (mx - mn)

    ax_chart = fig.add_axes([0.12, 0.12, 0.78, 0.6])
    ax_chart.set_facecolor(WHITE)

    y_pos = np.arange(len(tasks))
    bar_h = 0.25

    # Ceiling bars (background)
    ax_chart.barh(y_pos, ceilings, height=bar_h*2.5, color='#e2e8f0', edgecolor='#cbd5e0',
                  label='Practical Ceiling')
    # Our best
    ax_chart.barh(y_pos, our_best, height=bar_h*2.5, color=ACCENT_BLUE, edgecolor='#2b6cb0',
                  alpha=0.8, label='Our Best')
    # Baseline
    ax_chart.barh(y_pos, baselines, height=bar_h*2.5, color='#cbd5e0', edgecolor='#a0aec0',
                  alpha=0.7, label='Task Baseline')

    ax_chart.set_yticks(y_pos)
    ax_chart.set_yticklabels(tasks, fontsize=12, fontfamily=FONT)
    ax_chart.set_xlabel('Normalized Score', fontsize=11, fontfamily=FONT)
    ax_chart.legend(loc='lower right', fontsize=10)
    ax_chart.spines['top'].set_visible(False)
    ax_chart.spines['right'].set_visible(False)

    # Headroom annotations
    headrooms = [
        ('Mountain Car', '~25% headroom', ACCENT_RED),
        ('Battle of Sexes', '~10% headroom', '#b7791f'),
        ('Titanic', 'Near/past ceiling!', ACCENT_GREEN),
        ('Prisoner Dilemma', '~5% headroom', '#b7791f'),
        ('Regression', '~1% headroom (solved)', ACCENT_GREEN),
    ]
    for i, (task, hr, color) in enumerate(headrooms):
        ax.text(14.5, 4.7 - i * 0.85, f'{task}:', fontsize=9, fontweight='bold',
                color=TITLE_COLOR, fontfamily=FONT)
        ax.text(14.5, 4.35 - i * 0.85, hr, fontsize=9, color=color, fontfamily=FONT)

    return fig


def slide_17_voi_concept():
    fig, ax = new_slide()
    slide_title(ax, 'VoI-Guided RL: Value of Information',
                'A hypothesis is valuable if resolving it changes experimental strategy')
    add_footer(ax, 17)

    # Central hypothesis
    draw_box(ax, 4.5, 5.8, 7, 0.8, 'Hypothesis H: "Tree methods dominate on this dataset"',
             facecolor='#e9d8fd', edgecolor='#805ad5', fontsize=12, bold=True, linewidth=2)

    # Three branches
    draw_box(ax, 0.3, 3.8, 4.2, 1.0,
             'If H = TRUE:\nFocus on XGBoost, RF, LightGBM\nIgnore neural nets and SVMs',
             facecolor=GOOD_BG, edgecolor=ACCENT_GREEN, fontsize=10)
    draw_box(ax, 5.8, 3.8, 4.2, 1.0,
             'If H = FALSE:\nExplore neural nets, SVMs,\nensembles, stacking',
             facecolor=NODE_ORANGE, edgecolor='#dd6b20', fontsize=10)
    draw_box(ax, 11.3, 3.8, 4.2, 1.0,
             'If UNRESOLVED:\nSpread budget across everything\n(wasteful, unfocused)',
             facecolor='#e2e8f0', edgecolor=MUTED_COLOR, fontsize=10)

    draw_arrow(ax, 6.0, 5.8, 2.4, 4.8, color=ACCENT_GREEN, lw=2)
    draw_arrow(ax, 8.0, 5.8, 7.9, 4.8, color='#dd6b20', lw=2)
    draw_arrow(ax, 10.0, 5.8, 13.4, 4.8, color=MUTED_COLOR, lw=2)

    # VoI formula
    ax.add_patch(FancyBboxPatch((0.5, 1.8), 7.0, 1.5, boxstyle="round,pad=0.15",
                                 facecolor='#faf5ff', edgecolor='#805ad5', linewidth=2))
    ax.text(4.0, 3.05, 'VoI Computation', fontsize=13, fontweight='bold',
            color='#553c9a', ha='center', fontfamily=FONT)
    formula_lines = [
        'VoI(H) = KL( P_resolved || P_unresolved )',
        'P_resolved = p(H=T) * P_H_true + p(H=F) * P_H_false',
        'High VoI = resolving H shifts strategy significantly',
    ]
    for i, line in enumerate(formula_lines):
        ax.text(0.8, 2.6 - i * 0.32, line, fontsize=10, color='#553c9a',
                fontfamily='DejaVu Sans Mono' if i < 2 else FONT,
                fontstyle='italic' if i == 2 else 'normal')

    # Two types
    ax.add_patch(FancyBboxPatch((8.0, 1.8), 7.5, 1.5, boxstyle="round,pad=0.15",
                                 facecolor=LIGHT_GRAY, edgecolor=BORDER_GRAY, linewidth=1.5))
    ax.text(11.75, 3.05, 'Two Types of Valuable Hypotheses', fontsize=12, fontweight='bold',
            color=TITLE_COLOR, ha='center', fontfamily=FONT)
    ax.text(8.3, 2.55, '\u2022 Score-improving: "Tree methods dominate"', fontsize=10,
            color=ACCENT_GREEN, fontfamily=FONT)
    ax.text(8.6, 2.2, '\u2192 Focus on tree methods, improve best score', fontsize=9,
            color=BODY_COLOR, fontfamily=FONT)
    ax.text(8.3, 1.8, '\u2022 Space-pruning: "RNNs won\'t work here"', fontsize=10,
            color=ACCENT_BLUE, fontfamily=FONT)
    ax.text(8.6, 1.45, '\u2192 Eliminate model class, save budget', fontsize=9,
            color=BODY_COLOR, fontfamily=FONT)

    return fig


def slide_18_voi_example():
    fig, ax = new_slide()
    slide_title(ax, 'VoI: Worked Example (Titanic, Node 1)',
                'Computing information value for hypothesis-driven exploration')
    add_footer(ax, 18)

    # Hypothesis
    ax.add_patch(FancyBboxPatch((0.5, 5.5), 15.0, 1.5, boxstyle="round,pad=0.15",
                                 facecolor='#e9d8fd', edgecolor='#805ad5', linewidth=2))
    ax.text(8.0, 6.75, 'H1: "FamilySize = SibSp + Parch + 1 is a useful latent variable"',
            fontsize=13, fontweight='bold', color='#553c9a', ha='center', fontfamily=FONT)

    # Distributions
    cats = ['feat_eng', 'tree', 'preprocess', 'nn']
    p_unresolved = [0.35, 0.30, 0.20, 0.15]
    p_true = [0.55, 0.25, 0.15, 0.05]
    p_false = [0.20, 0.35, 0.25, 0.20]

    ax.text(0.8, 5.85, 'P_unresolved = {feat_eng: 0.35, tree: 0.30, preprocess: 0.20, nn: 0.15}',
            fontsize=9.5, color=MUTED_COLOR, fontfamily='DejaVu Sans Mono')
    ax.text(0.8, 5.55, 'P_H1_true    = {feat_eng: 0.55, tree: 0.25, preprocess: 0.15, nn: 0.05}    logit P(H1=true) = 0.52',
            fontsize=9.5, color=ACCENT_GREEN, fontfamily='DejaVu Sans Mono')

    # Mini bar chart for distributions
    ax_bars = fig.add_axes([0.06, 0.32, 0.42, 0.3])
    ax_bars.set_facecolor(WHITE)
    x = np.arange(len(cats))
    w = 0.25
    ax_bars.bar(x - w, p_unresolved, w, label='P_unresolved', color='#cbd5e0', edgecolor='#a0aec0')
    ax_bars.bar(x, p_true, w, label='P_H_true', color=ACCENT_GREEN, alpha=0.7, edgecolor='#276749')
    ax_bars.bar(x + w, p_false, w, label='P_H_false', color=ACCENT_RED, alpha=0.7, edgecolor='#9b2c2c')
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(cats, fontsize=9, fontfamily=FONT)
    ax_bars.legend(fontsize=8, loc='upper right')
    ax_bars.set_title('Experiment Strategy Distributions', fontsize=10, fontfamily=FONT)
    ax_bars.spines['top'].set_visible(False)
    ax_bars.spines['right'].set_visible(False)

    # Result and rewards
    ax.add_patch(FancyBboxPatch((8.0, 1.5), 7.5, 3.5, boxstyle="round,pad=0.15",
                                 facecolor=GOOD_BG, edgecolor=ACCENT_GREEN, linewidth=1.5))
    ax.text(11.75, 4.75, 'Experiment Result', fontsize=13, fontweight='bold',
            color=ACCENT_GREEN, ha='center', fontfamily=FONT)
    result_lines = [
        'Prediction: XGBoost + FamilySize > baseline by 0.02-0.04',
        'Result:     0.793 vs 0.764 = +0.029  (within prediction!)',
        '',
        'VoI(H1)    = KL(P_resolved || P_unresolved) = 0.18',
        '',
        'Rewards:',
        '  R1 (resolution)  = high (sharp, correct prediction)',
        '  R2 (information) = 0.18 (reasonably high VoI)',
        '  R3 (performance) = +0.029 / 0.764 = 3.8% improvement',
    ]
    for i, line in enumerate(result_lines):
        ax.text(8.3, 4.35 - i * 0.33, line, fontsize=9.5, color=BODY_COLOR,
                fontfamily='DejaVu Sans Mono' if i in [0,1,3] else FONT)

    return fig


def slide_19_reward_structure():
    fig, ax = new_slide()
    slide_title(ax, 'Reward Structure: R1 + R2 + R3',
                'Dense per-node + sparse end-of-tree signal')
    add_footer(ax, 19)

    # Three reward boxes
    rewards = [
        ('R1: Resolution', '0.4', ACCENT_BLUE, '#dbeafe', [
            'Did the experiment move confidence',
            'in its linked hypothesis?',
            '',
            'Measured by:',
            '\u2022 Prediction sharpness',
            '\u2022 Directional correctness',
            '',
            'Applied: per node (validate/challenge)',
            '',
            'Dense signal for credit assignment',
        ]),
        ('R2: Information (VoI)', '0.3', '#805ad5', '#e9d8fd', [
            'Did this experiment introduce a',
            'valuable structural claim?',
            '',
            'Measured by:',
            '\u2022 KL(P_resolved || P_unresolved)',
            '\u2022 High = strategy would change',
            '',
            'Applied: explore nodes only',
            '',
            'Rewards informative hypotheses',
        ]),
        ('R3: Performance', '0.3', ACCENT_GREEN, '#d1fae5', [
            'Did the tree find a good solution?',
            '',
            'Measured by:',
            '\u2022 Best normalized score at end',
            '\u2022 Improvement over task baseline',
            '',
            'Applied: once at end of tree',
            '',
            'Sparse but grounding signal',
            '(prevents endless exploration)',
        ]),
    ]

    for i, (title, weight, border, bg, lines) in enumerate(rewards):
        x = 0.5 + i * 5.2
        ax.add_patch(FancyBboxPatch((x, 1.5), 4.8, 5.5, boxstyle="round,pad=0.15",
                                     facecolor=bg, edgecolor=border, linewidth=2))
        ax.text(x + 2.4, 6.7, title, fontsize=13, fontweight='bold', color=border,
                ha='center', fontfamily=FONT)
        ax.text(x + 2.4, 6.25, f'weight = {weight}', fontsize=10, color=MUTED_COLOR,
                ha='center', fontfamily=FONT)
        for j, line in enumerate(lines):
            ax.text(x + 0.2, 5.8 - j * 0.38, line, fontsize=9.5, color=BODY_COLOR, fontfamily=FONT)

    # Formula at bottom
    ax.add_patch(FancyBboxPatch((2.0, 0.5), 12.0, 0.8, boxstyle="round,pad=0.1",
                                 facecolor=LIGHT_GRAY, edgecolor=BORDER_GRAY, linewidth=1.5))
    ax.text(8.0, 0.9, 'R_node = 0.4 * R1 + 0.3 * R2  (dense, per-node)     |     R_final = 0.3 * R3  (sparse, end-of-tree)',
            fontsize=11, fontweight='bold', color=TITLE_COLOR, ha='center', fontfamily='DejaVu Sans Mono')

    return fig


def slide_20_summary():
    fig, ax = new_slide()
    slide_title(ax, 'Summary & Next Steps')
    add_footer(ax, 20)

    # Key results (left)
    ax.add_patch(FancyBboxPatch((0.3, 2.0), 7.5, 5.2, boxstyle="round,pad=0.15",
                                 facecolor=GOOD_BG, edgecolor=ACCENT_BLUE, linewidth=1.5))
    ax.text(4.05, 6.9, 'Key Results', fontsize=15, fontweight='bold',
            color=ACCENT_BLUE, ha='center', fontfamily=FONT)

    results = [
        ('Architecture', 'Scientist+Executor > single-model MCTS'),
        ('Selection', 'LLM scientist > formula-based (UCB/softmax)'),
        ('Training data', 'Task-grounded >> Claude traces >> templates'),
        ('Training scope', 'Per-task >> multi-task (multi-task hurts)'),
        ('Best results', '+0.046 accuracy Titanic (p=0.032)'),
        ('N20 scaling', 'Titanic 0.827 \u2192 0.945 with more budget'),
        ('Ceilings', 'Regression solved, MountainCar has ~25% headroom'),
        ('Key insight', 'SFT teaches WHAT; RL needed to teach HOW'),
    ]
    for i, (key, val) in enumerate(results):
        y = 6.4 - i * 0.55
        ax.text(0.6, y, f'{key}:', fontsize=10, fontweight='bold', color=TITLE_COLOR, fontfamily=FONT)
        ax.text(3.3, y, val, fontsize=9.5, color=BODY_COLOR, fontfamily=FONT)

    # Next steps (right)
    ax.add_patch(FancyBboxPatch((8.3, 2.0), 7.2, 5.2, boxstyle="round,pad=0.15",
                                 facecolor='#faf5ff', edgecolor='#805ad5', linewidth=1.5))
    ax.text(11.9, 6.9, 'Next Steps', fontsize=15, fontweight='bold',
            color='#553c9a', ha='center', fontfamily=FONT)

    steps = [
        '1. Complete n20 evaluation across all tasks',
        '2. GRPO training with VoI rewards',
        '   \u2022 Improved validation/rejection logic',
        '   \u2022 W&B monitoring (voi-scientist-rl)',
        '3. RL task pipeline (Mountain Car priority)',
        '   \u2022 Build mlgym_rl.sif container',
        '   \u2022 ~25% headroom = most room to improve',
        '4. Ablation: contribution of each reward',
        '5. Scale: larger models (8B, 14B)',
        '6. Generalization: held-out MLGym tasks',
    ]
    for i, step in enumerate(steps):
        ax.text(8.6, 6.4 - i * 0.44, step, fontsize=10, color=BODY_COLOR, fontfamily=FONT)

    # Bottom tagline
    ax.add_patch(FancyBboxPatch((1.0, 0.5), 14.0, 1.0, boxstyle="round,pad=0.1",
                                 facecolor=TITLE_COLOR, edgecolor='none'))
    ax.text(8.0, 1.0, 'Goal: A scientist that finds better solutions faster, with fewer wasted experiments,\n'
            'by learning the structure of each problem rather than performing shallow local optimization.',
            fontsize=12, color=WHITE, ha='center', va='center', fontfamily=FONT, linespacing=1.4,
            fontstyle='italic')

    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    slides = [
        slide_01_title,
        slide_02_problem,
        slide_03_aira_problems,
        slide_04_more_mcts_problems,
        slide_05_our_approach_tree,
        slide_06_our_approach_flow,
        slide_07_battle_of_sexes,
        slide_08_hallucination_failure,
        slide_09_why_ttt,
        slide_10_sft_pipeline,
        slide_11_training_data,
        slide_12_grounding_fix,
        slide_13_sft_results,
        slide_14_in_vs_out_domain,
        slide_15_n20,
        slide_16_ceiling,
        slide_17_voi_concept,
        slide_18_voi_example,
        slide_19_reward_structure,
        slide_20_summary,
    ]

    print(f'Generating {len(slides)} slides...')
    with PdfPages(OUTPUT_PATH) as pdf:
        for i, slide_fn in enumerate(slides):
            print(f'  Slide {i+1}: {slide_fn.__name__}')
            fig = slide_fn()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f'\nDone! PDF saved to: {OUTPUT_PATH}')
    print(f'  {len(slides)} slides, 16:9 aspect ratio, 150 DPI')


if __name__ == '__main__':
    main()
