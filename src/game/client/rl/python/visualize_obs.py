#!/usr/bin/env python3
"""
Visualize Teeworlds RL Observations

Launches a server with bots and renders what the agent actually sees -- the
semantic tile grid, raycasts, hook state and the next checkpoint -- so you can
check an encoding against reality instead of trusting the struct layout.

Usage:
    python visualize_obs.py
    python visualize_obs.py --num-bots 4 --bot-id 2
    python visualize_obs.py --map Tutorial --scale 12

Controls:
    a/d or arrows: Move left/right
    space: Jump
    n/p: Next/previous bot
    r: Toggle raycasts
    c: Toggle checkpoint arrow
    q: Quit
"""

import argparse
import sys
import time
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from observation import (
    StructuredObservation, TileType,
    TILE_GRID_WIDTH, TILE_GRID_HEIGHT, NUM_RAYCASTS
)

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Terminal colors (ANSI escape codes)
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Tile colors
    BLACK = '\033[40m'      # Empty/air
    WHITE = '\033[47m'      # Solid
    RED = '\033[41m'        # Death
    YELLOW = '\033[43m'     # Nohook
    CYAN = '\033[46m'       # Freeze
    GREEN = '\033[42m'      # Unfreeze
    MAGENTA = '\033[45m'    # Teleport
    BLUE = '\033[44m'       # Boost/other special
    
    # Text colors
    FG_WHITE = '\033[97m'
    FG_BLACK = '\033[30m'
    FG_RED = '\033[91m'
    FG_GREEN = '\033[92m'
    FG_YELLOW = '\033[93m'


# Unicode characters for tiles
TILE_CHARS = {
    TileType.EMPTY: ' ',
    TileType.SOLID: '█',
    TileType.DEATH: '☠',
    TileType.NOHOOK: '○',
    TileType.FREEZE: '❄',
    TileType.UNFREEZE: '♨',
    TileType.TELEPORT: '◎',
    TileType.BOOST: '→',
    TileType.STOPPER: '■',
    TileType.SWITCH: '⚡',
    TileType.JUMP: '↑',
    TileType.THROUGH: '░',
    TileType.SPAWN: 'S',
    TileType.FINISH: 'F',
    TileType.CHECKPOINT: 'C',
    TileType.OTHER: '?',
}

# Simple ASCII fallback
TILE_CHARS_ASCII = {
    TileType.EMPTY: ' ',
    TileType.SOLID: '#',
    TileType.DEATH: 'X',
    TileType.NOHOOK: 'o',
    TileType.FREEZE: '*',
    TileType.UNFREEZE: '+',
    TileType.TELEPORT: '@',
    TileType.BOOST: '>',
    TileType.STOPPER: '=',
    TileType.SWITCH: '!',
    TileType.JUMP: '^',
    TileType.THROUGH: '.',
    TileType.SPAWN: 'S',
    TileType.FINISH: 'F',
    TileType.CHECKPOINT: 'C',
    TileType.OTHER: '?',
}

# Colors for each tile type
TILE_COLORS = {
    TileType.EMPTY: Colors.BLACK,
    TileType.SOLID: Colors.WHITE,
    TileType.DEATH: Colors.RED,
    TileType.NOHOOK: Colors.YELLOW,
    TileType.FREEZE: Colors.CYAN,
    TileType.UNFREEZE: Colors.GREEN,
    TileType.TELEPORT: Colors.MAGENTA,
    TileType.BOOST: Colors.BLUE,
    TileType.STOPPER: Colors.WHITE,
    TileType.SWITCH: Colors.MAGENTA,
    TileType.JUMP: Colors.GREEN,
    TileType.THROUGH: Colors.BLACK,
    TileType.SPAWN: Colors.GREEN,
    TileType.FINISH: Colors.GREEN,
    TileType.CHECKPOINT: Colors.CYAN,
    TileType.OTHER: Colors.BLUE,
}


def clear_screen():
    """Clear terminal screen."""
    print('\033[2J\033[H', end='')


def render_tile_grid_terminal(obs: StructuredObservation, use_color: bool = True, use_unicode: bool = True):
    """Render the tile grid in the terminal."""
    chars = TILE_CHARS if use_unicode else TILE_CHARS_ASCII
    
    lines = []
    
    # Header
    lines.append(f"{'─' * 60}")
    lines.append(f"Tick: {obs.tick:>8} | Players: {obs.num_active_players} | Projectiles: {obs.num_projectiles}")
    lines.append(f"Player Pos: ({obs.local_player.pos_x:.1f}, {obs.local_player.pos_y:.1f})")
    lines.append(f"Player Vel: ({obs.local_player.vel_x:.2f}, {obs.local_player.vel_y:.2f})")
    lines.append(f"Hook: {obs.local_player.hook_state} | Weapon: {obs.local_player.weapon} | Frozen: {obs.local_player.is_frozen}")
    lines.append(f"{'─' * 60}")
    
    # Top border
    lines.append('┌' + '─' * TILE_GRID_WIDTH + '┐')
    
    # Tile grid
    for y in range(TILE_GRID_HEIGHT):
        row = '│'
        for x in range(TILE_GRID_WIDTH):
            tile_val = obs.tile_grid[y, x]
            try:
                tile_type = TileType(tile_val)
            except ValueError:
                tile_type = TileType.OTHER
            
            char = chars.get(tile_type, '?')
            
            # Mark center (player position) with special character
            # Center is at (TILE_GRID_WIDTH // 2, TILE_GRID_HEIGHT // 2)
            if x == TILE_GRID_WIDTH // 2 and y == TILE_GRID_HEIGHT // 2:
                if use_color:
                    row += f"{Colors.FG_RED}{Colors.BOLD}@{Colors.RESET}"
                else:
                    row += '@'
            elif use_color:
                color = TILE_COLORS.get(tile_type, Colors.BLACK)
                # Use foreground color for character visibility
                fg = Colors.FG_BLACK if tile_type == TileType.SOLID else Colors.FG_WHITE
                row += f"{color}{fg}{char}{Colors.RESET}"
            else:
                row += char
        row += '│'
        lines.append(row)
    
    # Bottom border
    lines.append('└' + '─' * TILE_GRID_WIDTH + '┘')
    
    # Legend
    lines.append("")
    lines.append("Legend: " + " ".join([
        f"{chars[t]}={t.name[:3]}" for t in [
            TileType.SOLID, TileType.DEATH, TileType.FREEZE, 
            TileType.NOHOOK, TileType.TELEPORT, TileType.FINISH
        ]
    ]))
    lines.append("@ = Player position (center)")
    
    return '\n'.join(lines)


def render_to_image(obs: StructuredObservation, filename: str = "observation.png"):
    """Render the observation to a PNG image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Error: Pillow is required for image output. Install with: pip install Pillow")
        return False
    
    # Tile colors (RGB)
    tile_colors_rgb = {
        TileType.EMPTY: (40, 40, 50),       # Dark gray
        TileType.SOLID: (200, 200, 200),    # Light gray
        TileType.DEATH: (200, 50, 50),      # Red
        TileType.NOHOOK: (200, 200, 100),   # Yellow
        TileType.FREEZE: (100, 200, 255),   # Cyan
        TileType.UNFREEZE: (100, 255, 100), # Green
        TileType.TELEPORT: (200, 100, 255), # Magenta
        TileType.BOOST: (100, 100, 255),    # Blue
        TileType.STOPPER: (150, 150, 150),  # Gray
        TileType.SWITCH: (255, 100, 255),   # Pink
        TileType.JUMP: (100, 255, 150),     # Light green
        TileType.THROUGH: (80, 80, 80),     # Dark
        TileType.SPAWN: (50, 200, 50),      # Green
        TileType.FINISH: (50, 255, 50),     # Bright green
        TileType.CHECKPOINT: (100, 200, 200), # Teal
        TileType.OTHER: (100, 100, 100),    # Gray
    }
    
    tile_size = 16
    margin = 60
    info_height = 100
    
    img_width = TILE_GRID_WIDTH * tile_size + margin * 2
    img_height = TILE_GRID_HEIGHT * tile_size + margin * 2 + info_height
    
    img = Image.new('RGB', (img_width, img_height), (30, 30, 40))
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    # Draw info header
    y_offset = 10
    draw.text((margin, y_offset), f"Tick: {obs.tick}", fill=(255, 255, 255), font=font)
    y_offset += 18
    draw.text((margin, y_offset), 
              f"Pos: ({obs.local_player.pos_x:.1f}, {obs.local_player.pos_y:.1f})", 
              fill=(255, 255, 255), font=font)
    y_offset += 18
    draw.text((margin, y_offset), 
              f"Vel: ({obs.local_player.vel_x:.2f}, {obs.local_player.vel_y:.2f})", 
              fill=(200, 200, 200), font=font)
    y_offset += 18
    draw.text((margin, y_offset), 
              f"Hook: {obs.local_player.hook_state} | Weapon: {obs.local_player.weapon} | Frozen: {obs.local_player.is_frozen}", 
              fill=(200, 200, 200), font=font)
    
    # Draw tile grid
    grid_y_start = info_height + margin
    
    for y in range(TILE_GRID_HEIGHT):
        for x in range(TILE_GRID_WIDTH):
            tile_val = obs.tile_grid[y, x]
            try:
                tile_type = TileType(tile_val)
            except ValueError:
                tile_type = TileType.OTHER
            
            color = tile_colors_rgb.get(tile_type, (100, 100, 100))
            
            px = margin + x * tile_size
            py = grid_y_start + y * tile_size
            
            draw.rectangle([px, py, px + tile_size - 1, py + tile_size - 1], fill=color)
    
    # Draw player marker at center
    center_x = margin + (TILE_GRID_WIDTH // 2) * tile_size + tile_size // 2
    center_y = grid_y_start + (TILE_GRID_HEIGHT // 2) * tile_size + tile_size // 2
    radius = tile_size // 2 - 2
    draw.ellipse([center_x - radius, center_y - radius, 
                  center_x + radius, center_y + radius], 
                 fill=(255, 100, 100), outline=(255, 255, 255))
    
    # Draw other players (relative to local player)
    half_w = TILE_GRID_WIDTH // 2
    half_h = TILE_GRID_HEIGHT // 2
    for other in obs.other_players:
        if other.active:
            # Calculate relative tile position
            rel_x = (other.pos_x - obs.local_player.pos_x) / 32.0
            rel_y = (other.pos_y - obs.local_player.pos_y) / 32.0

            # Check if within grid bounds
            if -half_w <= rel_x <= half_w and -half_h <= rel_y <= half_h:
                px = margin + int(half_w + rel_x) * tile_size + tile_size // 2
                py = grid_y_start + int(half_h + rel_y) * tile_size + tile_size // 2
                draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                            fill=(100, 100, 255), outline=(255, 255, 255))
    
    # Draw legend
    legend_y = grid_y_start + TILE_GRID_HEIGHT * tile_size + 10
    legend_items = [
        (TileType.SOLID, "Solid"),
        (TileType.DEATH, "Death"),
        (TileType.FREEZE, "Freeze"),
        (TileType.NOHOOK, "NoHook"),
        (TileType.TELEPORT, "Tele"),
        (TileType.FINISH, "Finish"),
    ]
    
    x_offset = margin
    for tile_type, name in legend_items:
        color = tile_colors_rgb[tile_type]
        draw.rectangle([x_offset, legend_y, x_offset + 12, legend_y + 12], fill=color)
        draw.text((x_offset + 16, legend_y), name, fill=(200, 200, 200), font=font_small)
        x_offset += 70
    
    img.save(filename)
    print(f"Saved observation to {filename}")
    return True


def render_cv2_visualization(obs: StructuredObservation, scale: int = 8,
                              show_raycasts: bool = True, show_checkpoint: bool = True):
    """
    Render observation as a CV2 image with raycasts and checkpoint arrow.

    Args:
        obs: Structured observation
        scale: Pixels per tile
        show_raycasts: Draw raycast beams
        show_checkpoint: Draw checkpoint direction arrow

    Returns:
        BGR image (numpy array)
    """
    if not HAS_CV2:
        return None

    # Tile colors (BGR for OpenCV)
    tile_colors_bgr = {
        TileType.EMPTY: (40, 40, 40),
        TileType.SOLID: (200, 200, 200),
        TileType.DEATH: (50, 50, 200),       # Red
        TileType.NOHOOK: (100, 150, 150),
        TileType.FREEZE: (200, 150, 50),     # Blue-ish
        TileType.UNFREEZE: (50, 200, 100),
        TileType.TELEPORT: (200, 50, 200),   # Magenta
        TileType.BOOST: (200, 200, 50),      # Cyan
        TileType.STOPPER: (80, 80, 100),
        TileType.SWITCH: (50, 150, 150),
        TileType.JUMP: (50, 255, 50),        # Bright green
        TileType.THROUGH: (100, 100, 100),
        TileType.SPAWN: (50, 200, 200),      # Yellow
        TileType.FINISH: (50, 255, 255),     # Yellow
        TileType.CHECKPOINT: (50, 165, 255), # Orange
        TileType.OTHER: (100, 100, 100),
    }

    h, w = TILE_GRID_HEIGHT, TILE_GRID_WIDTH
    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

    # Draw tile grid
    for y in range(h):
        for x in range(w):
            tile_val = obs.tile_grid[y, x]
            try:
                tile_type = TileType(tile_val)
            except ValueError:
                tile_type = TileType.OTHER
            color = tile_colors_bgr.get(tile_type, (100, 100, 100))

            y1, y2 = y * scale, (y + 1) * scale
            x1, x2 = x * scale, (x + 1) * scale
            img[y1:y2, x1:x2] = color

    # Player center in pixels
    cx = (TILE_GRID_WIDTH // 2) * scale + scale // 2
    cy = (TILE_GRID_HEIGHT // 2) * scale + scale // 2

    # Draw raycasts
    if show_raycasts:
        max_ray_dist = 20 * scale  # 20 tiles
        for i in range(NUM_RAYCASTS):
            angle = i * (2 * np.pi / NUM_RAYCASTS)
            dir_x = np.cos(angle)
            dir_y = np.sin(angle)

            # Solid raycast
            solid_dist = obs.raycast_solid[i] * max_ray_dist
            end_x = int(cx + dir_x * solid_dist)
            end_y = int(cy + dir_y * solid_dist)
            cv2.line(img, (cx, cy), (end_x, end_y), (100, 100, 100), 1)
            cv2.circle(img, (end_x, end_y), 2, (200, 200, 200), -1)

    # Draw checkpoint arrow
    if show_checkpoint and obs.total_checkpoints > 0:
        cp_dx = obs.next_checkpoint_x - obs.local_player.pos_x
        cp_dy = obs.next_checkpoint_y - obs.local_player.pos_y
        dist = np.sqrt(cp_dx * cp_dx + cp_dy * cp_dy)
        if dist > 1:
            arrow_len = min(60, dist / 16) * scale // 8
            dir_x = cp_dx / dist
            dir_y = cp_dy / dist
            end_x = int(cx + dir_x * arrow_len)
            end_y = int(cy + dir_y * arrow_len)
            cv2.arrowedLine(img, (cx, cy), (end_x, end_y), (0, 165, 255), 3, tipLength=0.3)

    # Draw velocity vector
    vel_scale = 2.0 * scale / 8
    vel_x = int(obs.local_player.vel_x * vel_scale)
    vel_y = int(obs.local_player.vel_y * vel_scale)
    if abs(vel_x) > 1 or abs(vel_y) > 1:
        cv2.arrowedLine(img, (cx, cy), (cx + vel_x, cy + vel_y), (255, 255, 255), 2)

    # Draw player circle
    cv2.circle(img, (cx, cy), scale // 2, (50, 255, 50), -1)
    cv2.circle(img, (cx, cy), scale // 2, (255, 255, 255), 1)

    # Add info panel at top
    panel_height = 100
    padded = np.zeros((img.shape[0] + panel_height, img.shape[1], 3), dtype=np.uint8)
    padded[panel_height:] = img

    # Draw text info
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (255, 255, 255)
    p = obs.local_player

    lines = [
        f"Tick: {obs.tick}  Pos: ({p.pos_x:.0f}, {p.pos_y:.0f})",
        f"Vel: ({p.vel_x:.1f}, {p.vel_y:.1f})  Speed: {np.sqrt(p.vel_x**2 + p.vel_y**2):.1f}",
        f"Grounded: {obs.is_grounded}  Jumps: {obs.jumps_remaining}  Hook: {p.hook_state}",
        f"Frozen: {p.is_frozen}  Checkpoint: {obs.checkpoint_index}/{obs.total_checkpoints}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(padded, line, (5, 15 + i * 20), font, font_scale, color, 1)

    return padded


def run_server_side_visualization(num_bots: int = 1, bot_id: int = 0, scale: int = 10,
                                   fps_limit: float = 30.0, map_name: str = "Tutorial"):
    """Run live OpenCV visualization for server-side bots."""
    if not HAS_CV2:
        print("OpenCV not available. Install with: pip install opencv-python")
        return

    from teeworlds_server_env import TeeWorldsServerEnv, ServerEnvConfig

    config = ServerEnvConfig(max_episode_steps=10000, map_name=map_name)
    env = TeeWorldsServerEnv(num_bots=num_bots, config=config)
    env.reset()

    print(f"Server started with {num_bots} bots. Visualizing bot {bot_id}.")
    print("Press 'q' to quit, 'r' to toggle raycasts, 'c' to toggle checkpoint, 'space' to jump.")

    show_raycasts = True
    show_checkpoint = True
    frame_time = 1.0 / fps_limit

    # Manual control state
    direction = 0
    jump = 0

    try:
        while True:
            start = time.time()

            # Get observation from server
            observations, rewards, dones = env.shm_client.read_observations_batch()
            obs = observations[bot_id] if bot_id < len(observations) else None

            if obs is None:
                time.sleep(0.01)
                continue

            img = render_cv2_visualization(obs, scale, show_raycasts, show_checkpoint)
            if img is not None:
                # Add reward info
                cv2.putText(img, f"Reward: {rewards[bot_id]:.4f}  Bot: {bot_id}/{num_bots}",
                           (5, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.imshow('DDNet RL Observation (Server-Side)', img)

            # Send action
            discrete = np.zeros((num_bots, 3), dtype=np.int32)
            discrete[bot_id, 0] = direction + 1  # 0=left, 1=none, 2=right
            discrete[bot_id, 1] = jump
            continuous = np.zeros((num_bots, 2), dtype=np.float32)
            env.shm_client.write_actions_batch({'discrete': discrete, 'continuous': continuous})
            env.shm_client.signal_actions_ready()
            env.shm_client.wait_for_observations(timeout=0.1)

            # Reset jump after sending
            jump = 0

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                show_raycasts = not show_raycasts
                print(f"Raycasts: {'ON' if show_raycasts else 'OFF'}")
            elif key == ord('c'):
                show_checkpoint = not show_checkpoint
                print(f"Checkpoint: {'ON' if show_checkpoint else 'OFF'}")
            elif key == ord('a') or key == 81:  # Left arrow
                direction = -1
            elif key == ord('d') or key == 83:  # Right arrow
                direction = 1
            elif key == ord(' '):  # Space = jump
                jump = 1
            elif key == ord('s') or key == 84:  # Down arrow / stop
                direction = 0
            elif key == ord('n'):  # Next bot
                bot_id = (bot_id + 1) % num_bots
                print(f"Now viewing bot {bot_id}")
            elif key == ord('p'):  # Previous bot
                bot_id = (bot_id - 1) % num_bots
                print(f"Now viewing bot {bot_id}")

            elapsed = time.time() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        env.close()
        if HAS_CV2:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualize Teeworlds RL observations")
    parser.add_argument('--scale', type=int, default=10, help='Pixels per tile')
    parser.add_argument('--fps', type=float, default=30, help='Refresh rate')
    parser.add_argument('--num-bots', type=int, default=1, help='Number of bots to launch')
    parser.add_argument('--bot-id', type=int, default=0, help='Which bot to visualize')
    parser.add_argument('--map', type=str, default='Tutorial', help='Map to load')
    # Accepted so existing commands keep working; server-side is the only mode now.
    parser.add_argument('--server-side', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    run_server_side_visualization(
        num_bots=args.num_bots,
        bot_id=args.bot_id,
        scale=args.scale,
        fps_limit=args.fps,
        map_name=args.map
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())

