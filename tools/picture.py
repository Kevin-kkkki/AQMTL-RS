import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, ConnectionPatch, Polygon

# ================= Configuration =================
# Professional Color Palette (IEEE / Science Style)
COLORS = {
    'bg_main': '#F8F9FA',      # Light Gray Background
    'input_box': '#E3F2FD',    # Light Blue
    'encoder_bg': '#E8EAF6',   # Light Indigo Background
    'layer_box': '#FFFFFF',    # White for layers
    'heads_box': '#F3E5F5',    # Light Purple
    'tsqp_header': '#1565C0',  # Dark Blue
    'tsqp_bg': '#E3F2FD',      # Light Blue
    'samp_header': '#C62828',  # Dark Red
    'samp_bg': '#FFEBEE',      # Light Red
    'text': '#263238',         # Dark Slate
    'arrow': '#455A64',        # Dark Gray Arrow
    'quant8': '#81D4FA',       # Cyan
    'quant4': '#E0E0E0',       # Light Gray
    'op_box': '#D1C4E9'        # Lavender
}

def draw_shadow_box(ax, xy, width, height, text, color='white', edge_color='#546E7A', fontsize=10, 
                    boxstyle="round,pad=0.1", alpha=1.0, fontweight='normal', shadow_offset=0.3, zorder=10):
    # Shadow
    shadow = FancyBboxPatch((xy[0]+shadow_offset, xy[1]-shadow_offset), width, height, 
                           boxstyle=boxstyle, ec='none', fc='#B0BEC5', zorder=zorder-1, alpha=0.5)
    ax.add_patch(shadow)
    
    # Main Box
    box = FancyBboxPatch(xy, width, height, boxstyle=boxstyle, ec=edge_color, fc=color, zorder=zorder, alpha=alpha, lw=1.2)
    ax.add_patch(box)
    
    # Text
    ax.text(xy[0] + width/2, xy[1] + height/2, text, ha='center', va='center', 
            fontsize=fontsize, fontweight=fontweight, color=COLORS['text'], zorder=zorder+1)
    return box

def draw_arrow(ax, xyA, xyB, color=COLORS['arrow'], arrowstyle='->', linestyle='-', lw=1.5, zorder=5):
    ax.annotate('', xy=xyB, xytext=xyA,
                arrowprops=dict(arrowstyle=arrowstyle, color=color, lw=lw, linestyle=linestyle), zorder=zorder)

fig, ax = plt.subplots(figsize=(16, 12)) # Slightly taller
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# ================= 1. Main Pipeline (Bottom) =================
# Background Container
pipeline_bg = FancyBboxPatch((2, 2), 96, 38, boxstyle="round,pad=0.2", ec='#CFD8DC', fc=COLORS['bg_main'], zorder=0, linestyle='--')
ax.add_patch(pipeline_bg)
ax.text(5, 37, "Main Multi-Task Pipeline", fontsize=12, fontweight='bold', color='#607D8B')

# Input
draw_shadow_box(ax, (4, 15), 10, 12, "Input\nImage", color=COLORS['input_box'], fontweight='bold')

# --- Shared Encoder (Now with Internal Structure) ---
# Encoder Background
encoder_frame = FancyBboxPatch((22, 5), 32, 32, boxstyle="round,pad=0.2", ec='#3F51B5', fc=COLORS['encoder_bg'], zorder=1, linestyle='-')
ax.add_patch(encoder_frame)
ax.text(38, 38.5, "Shared Encoder", fontsize=12, fontweight='bold', color='#3F51B5', ha='center', zorder=2)

# Internal Transformer Layers (Stacked)
# Layer 1
draw_shadow_box(ax, (26, 7), 24, 6, "Transformer Layer 1", color=COLORS['layer_box'], fontsize=9, zorder=5)
# Dots
ax.text(38, 15, "...", fontsize=14, fontweight='bold', ha='center', color='#5C6BC0', zorder=5)
# Layer i (The one being magnified)
target_layer = draw_shadow_box(ax, (26, 18), 24, 6, "Transformer Layer $i$\n(MSA + FFN)", color='#FFF9C4', fontsize=9, fontweight='bold', zorder=5)
# Layer N
draw_shadow_box(ax, (26, 28), 24, 6, "Transformer Layer N", color=COLORS['layer_box'], fontsize=9, zorder=5)

# Arrows inside Encoder
draw_arrow(ax, (38, 13), (38, 14.5), color='#5C6BC0')
draw_arrow(ax, (38, 16.5), (38, 18), color='#5C6BC0')
draw_arrow(ax, (38, 24), (38, 28), color='#5C6BC0')

# --- Task Heads ---
draw_shadow_box(ax, (65, 28), 12, 6, "Cls Head", color=COLORS['heads_box'])
draw_shadow_box(ax, (65, 18), 12, 6, "Det Head", color=COLORS['heads_box'])
draw_shadow_box(ax, (65, 8), 12, 6, "Seg Head", color=COLORS['heads_box'])

# Outputs
ax.text(85, 31, "Scene Class", fontsize=9, color=COLORS['text'])
ax.text(85, 21, "BBox Preds", fontsize=9, color=COLORS['text'])
ax.text(85, 11, "Seg Mask", fontsize=9, color=COLORS['text'])

# Main Connection Arrows
draw_arrow(ax, (14, 21), (22, 21), lw=2) # Input -> Enc
draw_arrow(ax, (54, 21), (65, 31), lw=1.5)
draw_arrow(ax, (54, 21), (65, 21), lw=1.5)
draw_arrow(ax, (54, 21), (65, 11), lw=1.5)
draw_arrow(ax, (77, 31), (83, 31))
draw_arrow(ax, (77, 21), (83, 21))
draw_arrow(ax, (77, 11), (83, 11))

# ================= 2. Detailed View Connection (Zoom Effect) =================
# Draw connecting lines from "Layer i" to the top modules
# Left Connector (TSQP)
con_poly_l = Polygon([(26, 24), (50, 24), (46, 53), (14, 53)], closed=True, fc=COLORS['tsqp_bg'], alpha=0.2, zorder=0)
ax.add_patch(con_poly_l)
# Right Connector (SAMP)
con_poly_r = Polygon([(26, 18), (50, 18), (86, 53), (54, 53)], closed=True, fc=COLORS['samp_bg'], alpha=0.2, zorder=0)
ax.add_patch(con_poly_r)

# ================= 3. Module A: TSQP (Top Left) =================
# Sub-region Background
tsqp_bg = FancyBboxPatch((14, 53), 32, 38, boxstyle="round,pad=0.2", ec='none', fc=COLORS['tsqp_bg'], alpha=0.3, zorder=0)
ax.add_patch(tsqp_bg)
ax.text(30, 88, "Module A: TSQP (Activation)", fontsize=11, fontweight='bold', color=COLORS['tsqp_header'], ha='center')

# Controller
draw_shadow_box(ax, (16, 70), 8, 6, "Task ID\n(k)", color='#FFF9C4', fontweight='bold')
# Switch
draw_shadow_box(ax, (28, 70), 6, 6, "Switch", color='white', boxstyle="circle")
draw_arrow(ax, (24, 73), (28, 73))

# Params Bank
draw_shadow_box(ax, (38, 78), 6, 4, "α_cls", color='white')
draw_shadow_box(ax, (38, 72), 6, 4, "α_det", color='white')
draw_shadow_box(ax, (38, 66), 6, 4, "α_seg", color='white')

# Dashed arrows to params
draw_arrow(ax, (34, 73), (38, 80), linestyle='--', color=COLORS['tsqp_header'])
draw_arrow(ax, (34, 73), (38, 74), linestyle='--', color=COLORS['tsqp_header'])
draw_arrow(ax, (34, 73), (38, 68), linestyle='--', color=COLORS['tsqp_header'])

# Selected Alpha
alpha_box = draw_shadow_box(ax, (25, 56), 12, 8, "Selected\nParameter\n(α_task)", color='#BBDEFB', fontweight='bold')

# Arrows converging
draw_arrow(ax, (44, 80), (37, 60), color=COLORS['tsqp_header'])
draw_arrow(ax, (44, 74), (37, 62), color=COLORS['tsqp_header'])
draw_arrow(ax, (44, 68), (37, 64), color=COLORS['tsqp_header'])

# Final arrow pointing down to Layer i
draw_arrow(ax, (31, 56), (32, 24), lw=2, linestyle='dotted', color=COLORS['tsqp_header'])
ax.text(33, 48, "Inject Scale", fontsize=9, color=COLORS['tsqp_header'], rotation=90)


# ================= 4. Module B: SAMP (Top Right) =================
# Sub-region Background
samp_bg = FancyBboxPatch((54, 53), 32, 38, boxstyle="round,pad=0.2", ec='none', fc=COLORS['samp_bg'], alpha=0.3, zorder=0)
ax.add_patch(samp_bg)
ax.text(70, 88, "Module B: SAMP (Weights)", fontsize=11, fontweight='bold', color=COLORS['samp_header'], ha='center')

# Inputs
draw_shadow_box(ax, (56, 76), 6, 6, "W", color='white', fontweight='bold')
draw_shadow_box(ax, (56, 68), 6, 6, "G", color='white', fontweight='bold')

# Score
draw_shadow_box(ax, (66, 72), 10, 8, "Score\n|W|*|G|", color='#FFE0B2')
draw_arrow(ax, (62, 79), (66, 78))
draw_arrow(ax, (62, 71), (66, 74))

# Mask
draw_shadow_box(ax, (78, 72), 6, 8, "Top-K\nMask", color='#FFD54F', fontweight='bold')
draw_arrow(ax, (76, 76), (78, 76))

# Quantizers
draw_shadow_box(ax, (60, 56), 8, 6, "8-bit\nHigh", color=COLORS['quant8'])
draw_shadow_box(ax, (76, 56), 8, 6, "4-bit\nLow", color=COLORS['quant4'])

# Mask logic arrows
draw_arrow(ax, (81, 72), (64, 62), linestyle='-')
ax.text(70, 66, "M=1", fontsize=8, color=COLORS['samp_header'], fontweight='bold')
draw_arrow(ax, (81, 72), (80, 62), linestyle='-')
ax.text(82, 66, "M=0", fontsize=8, color='#757575')

# Final arrow pointing down to Layer i
draw_arrow(ax, (70, 56), (45, 24), lw=2, linestyle='dotted', color=COLORS['samp_header'])
ax.text(62, 45, "Apply Mixed-P", fontsize=9, color=COLORS['samp_header'], rotation=-45)


plt.tight_layout()
plt.savefig('CTAQ_Framework_FullDetail.png', dpi=300, bbox_inches='tight')
plt.show()