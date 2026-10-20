# Future Improvements

## Action Space
- **Remove aim magnitude**: The continuous action space currently predicts (aim_angle, aim_magnitude) as two floats, but DDNet only uses the direction of `m_TargetX/m_TargetY` — magnitude is ignored by the game. Should be reduced to `continuous_dim: 1` (angle only). This wastes network capacity and adds a useless exploration dimension.
