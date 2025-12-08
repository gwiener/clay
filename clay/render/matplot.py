import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from clay import graph


def render(layout: graph.Layout, output_path: str, dpi: int = 72, diagnose: bool = False) -> None:
    """
    Render a graph layout to a PNG image using matplotlib.

    Args:
        layout: Layout object with graph and position information
        output_path: Path where the PNG file will be saved
        dpi: Dots per inch for the output image (default: 72)
        diagnose: If True, add diagnostic annotations (edge lengths, canvas frame)

    The function ensures 1:1 pixel mapping where each data point equals 1 pixel,
    and final image dimensions match the graph canvas size exactly.
    """
    graph = layout.graph
    width_px, height_px = graph.canvas.width, graph.canvas.height
    width_inches = width_px / dpi
    height_inches = height_px / dpi
    
    # Create figure with exact dimensions
    fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
    
    # Add axes that fill the entire figure (no margins)
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Turn off axes
    ax.set_axis_off()
    
    # Set exact data limits to match pixel dimensions
    ax.set_xlim(0, width_px)
    ax.set_ylim(0, height_px)

    # Draw canvas frame in diagnostic mode (inset by 1 to make all edges visible)
    if diagnose:
        frame = mpatches.Rectangle(
            (1, 1), width_px - 2, height_px - 2,
            linewidth=1, edgecolor='gray', facecolor='none'
        )
        ax.add_patch(frame)

    def _boundary_points(src_c, dst_c, src_node, dst_node):
        dx = dst_c[0] - src_c[0]
        dy = dst_c[1] - src_c[1]
        if dx == 0 and dy == 0:
            return src_c, dst_c
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        # Source
        if dx == 0:
            t_src = src_node.height / 2 / abs_dy
        elif dy == 0:
            t_src = src_node.width / 2 / abs_dx
        else:
            t_src = min(src_node.width / 2 / abs_dx, src_node.height / 2 / abs_dy)
        # Dest
        if dx == 0:
            t_dst = dst_node.height / 2 / abs_dy
        elif dy == 0:
            t_dst = dst_node.width / 2 / abs_dx
        else:
            t_dst = min(dst_node.width / 2 / abs_dx, dst_node.height / 2 / abs_dy)
        dir_vec = np.array([dx, dy])
        src_pt = np.array(src_c) + dir_vec * t_src
        dst_pt = np.array(dst_c) - dir_vec * t_dst
        return tuple(src_pt), tuple(dst_pt)

    # Draw edges (arrows)
    for src_name, dst_name in graph.edges:
        src_c = layout.get_node_center(src_name)
        dst_c = layout.get_node_center(dst_name)
        src_node = graph.nodes[graph.name2idx[src_name]]
        dst_node = graph.nodes[graph.name2idx[dst_name]]
        start_pt, end_pt = _boundary_points(src_c, dst_c, src_node, dst_node)
        ax.annotate(
            '',
            xy=end_pt,
            xytext=start_pt,
            arrowprops=dict(arrowstyle='->', lw=1, color='black', mutation_scale=20)
        )

        # Add edge length label in diagnostic mode
        if diagnose:
            length = ((end_pt[0] - start_pt[0])**2 + (end_pt[1] - start_pt[1])**2)**0.5
            mid_x = (start_pt[0] + end_pt[0]) / 2
            mid_y = (start_pt[1] + end_pt[1]) / 2
            ax.text(mid_x, mid_y, f"{length:.0f}", fontsize=8, color='gray', ha='center', va='center')

    # Draw nodes (boxes)
    for node in graph.nodes:
        cx, cy = layout.get_node_center(node.name)
        
        # Calculate top-left corner from center
        x = cx - node.width / 2
        y = cy - node.height / 2
        
        # Create rectangle
        rect = mpatches.Rectangle(
            (x, y), node.width, node.height,
            linewidth=1, edgecolor='black', facecolor='white'
        )
        ax.add_patch(rect)
        
        # Add text label centered in box
        ax.text(
            cx, cy,
            node.name,
            ha='center', va='center',
            fontsize=graph.defaults.font_size,
            fontname=graph.defaults.font_name
        )
    
    # Save with no padding
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0,
        format='png'
    )
    
    plt.close(fig)
