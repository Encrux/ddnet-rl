#!/usr/bin/env python3
"""
Visualize Teeworlds RL Observations

Renders the structured observation (tile grid, player state, etc.) 
in the terminal or as an image for debugging.

Usage:
    python visualize_obs.py              # Terminal visualization (continuous)
    python visualize_obs.py --once       # Single frame
    python visualize_obs.py --image      # Save as PNG image
    python visualize_obs.py --env-id 1   # Use different env ID
"""

import argparse
import sys
import time
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from teeworlds_env import (
    TeeWorldsShmClient, StructuredObservation, TileType,
    TILE_GRID_WIDTH, TILE_GRID_HEIGHT
)

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
            if x == 14 and y == 10:
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
    center_x = margin + 14 * tile_size + tile_size // 2
    center_y = grid_y_start + 10 * tile_size + tile_size // 2
    radius = tile_size // 2 - 2
    draw.ellipse([center_x - radius, center_y - radius, 
                  center_x + radius, center_y + radius], 
                 fill=(255, 100, 100), outline=(255, 255, 255))
    
    # Draw other players (relative to local player)
    for other in obs.other_players:
        if other.active:
            # Calculate relative tile position
            rel_x = (other.pos_x - obs.local_player.pos_x) / 32.0
            rel_y = (other.pos_y - obs.local_player.pos_y) / 32.0
            
            # Check if within grid bounds
            if -14 <= rel_x <= 14 and -10 <= rel_y <= 10:
                px = margin + int(14 + rel_x) * tile_size + tile_size // 2
                py = grid_y_start + int(10 + rel_y) * tile_size + tile_size // 2
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


def main():
    parser = argparse.ArgumentParser(description="Visualize Teeworlds RL observations")
    parser.add_argument('--env-id', type=int, default=0, help='Environment ID')
    parser.add_argument('--once', action='store_true', help='Show single frame and exit')
    parser.add_argument('--image', action='store_true', help='Save as PNG image')
    parser.add_argument('--no-color', action='store_true', help='Disable terminal colors')
    parser.add_argument('--ascii', action='store_true', help='Use ASCII instead of Unicode')
    parser.add_argument('--fps', type=float, default=10, help='Refresh rate for continuous mode')
    args = parser.parse_args()
    
    print(f"Connecting to shared memory (env_id={args.env_id})...")
    
    client = TeeWorldsShmClient(env_id=args.env_id, timeout=10.0)
    
    if not client.connect():
        print("Failed to connect to shared memory. Is the game running with rl_enable?")
        return 1
    
    print("Connected! Waiting for observations...")
    
    try:
        frame_count = 0
        while True:
            obs, info = client.wait_for_observation(timeout=5.0)
            
            if obs is None:
                print("Timeout waiting for observation")
                if args.once:
                    return 1
                continue
            
            if not isinstance(obs, StructuredObservation):
                print("Error: Expected structured observation, got pixel frame")
                print("Make sure the game is running with: rl_enable state")
                return 1
            
            frame_count += 1
            
            if args.image:
                render_to_image(obs, f"observation_{frame_count:04d}.png")
                if args.once:
                    break
            else:
                if not args.once:
                    clear_screen()
                
                output = render_tile_grid_terminal(
                    obs, 
                    use_color=not args.no_color,
                    use_unicode=not args.ascii
                )
                print(output)
                
                if args.once:
                    break
            
            # Send dummy action to keep the game running
            client.send_action(direction=0, jump=0, hook=0, fire=0)
            
            if not args.once:
                time.sleep(1.0 / args.fps)
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        client.disconnect()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
