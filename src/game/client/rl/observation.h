/* Teeworlds RL Bot - Structured Observation Definition */
#ifndef GAME_CLIENT_RL_OBSERVATION_H
#define GAME_CLIENT_RL_OBSERVATION_H

#include <cstdint>

namespace rl {

// Maximum number of other players to track
constexpr int MAX_OTHER_PLAYERS = 15;
// Maximum number of projectiles to track
constexpr int MAX_PROJECTILES = 32;
// Maximum number of pickups to track
constexpr int MAX_PICKUPS = 16;

// Local tile grid size (matches camera view)
constexpr int TILE_GRID_WIDTH = 29;   // 14 left + 1 center + 14 right
constexpr int TILE_GRID_HEIGHT = 21;  // 10 up + 1 center + 10 down
constexpr int TILE_GRID_SIZE = TILE_GRID_WIDTH * TILE_GRID_HEIGHT; // 609

// Tile types for the local grid
enum class ETileType : uint8_t
{
	EMPTY = 0,
	SOLID = 1,       // Wall/ground - blocks movement
	DEATH = 2,       // Instant death
	NOHOOK = 3,      // Can't hook
	FREEZE = 4,      // Freezes player
	UNFREEZE = 5,    // Unfreezes player
	TELEPORT = 6,    // Teleporter
	BOOST = 7,       // Speed boost
	STOPPER = 8,     // Stops movement
	SWITCH = 9,      // Switch tile
	JUMP = 10,       // Adds jump
	THROUGH = 11,    // One-way platform
	SPAWN = 12,      // Spawn point
	FINISH = 13,     // Finish line
	CHECKPOINT = 14, // Checkpoint
	OTHER = 15,      // Other special tiles
};

// Observation mode
enum class EObservationMode : uint8_t
{
	RENDERED = 0,      // RGB pixel frames (requires display)
	STRUCTURED = 1,    // Game state vectors (headless compatible)
};

// Structured observation for a single player
#pragma pack(push, 1)
struct SPlayerObs
{
	// Position and velocity (normalized to map bounds)
	float m_PosX;           // X position
	float m_PosY;           // Y position  
	float m_VelX;           // X velocity
	float m_VelY;           // Y velocity
	
	// State
	int8_t m_Direction;     // -1, 0, 1
	uint8_t m_Jumped;       // Jump state flags
	uint8_t m_HookState;    // Hook state (0=idle, 1=flying, 2=grabbed)
	uint8_t m_Weapon;       // Current weapon (0-5)
	
	// Hook info
	float m_HookPosX;       // Hook position X
	float m_HookPosY;       // Hook position Y
	int8_t m_HookedPlayer;  // -1 if not hooked to player
	
	// Flags
	uint8_t m_Active;       // Is this player active?
	uint8_t m_IsFrozen;     // DDNet freeze state
	uint8_t m_IsLocal;      // Is this the local player?
	
	// Health/Armor
	uint8_t m_Health;
	uint8_t m_Armor;
	
	uint8_t m_Padding[2];   // Align to 4 bytes
};
#pragma pack(pop)

static_assert(sizeof(SPlayerObs) == 36, "SPlayerObs must be 36 bytes");

// Structured observation for a projectile
#pragma pack(push, 1)
struct SProjectileObs
{
	float m_PosX;
	float m_PosY;
	float m_VelX;
	float m_VelY;
	uint8_t m_Type;         // Weapon type
	uint8_t m_Active;
	uint8_t m_Padding[2];
};
#pragma pack(pop)

static_assert(sizeof(SProjectileObs) == 20, "SProjectileObs must be 20 bytes");

// Structured observation for a pickup
#pragma pack(push, 1)
struct SPickupObs
{
	float m_PosX;
	float m_PosY;
	uint8_t m_Type;         // Pickup type
	uint8_t m_Active;
	uint8_t m_Padding[2];
};
#pragma pack(pop)

static_assert(sizeof(SPickupObs) == 12, "SPickupObs must be 12 bytes");

// Full structured observation
// Total size should be reasonable for shared memory
#pragma pack(push, 1)
struct SStructuredObservation
{
	// Game state (8 bytes)
	uint32_t m_Tick;
	uint8_t m_GameFlags;       // Team game, CTF, etc.
	uint8_t m_GameState;       // Paused, warmup, etc.
	uint8_t m_NumActivePlayers;
	uint8_t m_NumProjectiles;
	
	// Local player (36 bytes)
	SPlayerObs m_LocalPlayer;
	
	// Other players (36 * 15 = 540 bytes)
	SPlayerObs m_aOtherPlayers[MAX_OTHER_PLAYERS];
	
	// Projectiles (20 * 32 = 640 bytes)
	SProjectileObs m_aProjectiles[MAX_PROJECTILES];
	
	// Pickups (12 * 16 = 192 bytes)
	SPickupObs m_aPickups[MAX_PICKUPS];
	
	// Flags for CTF (24 bytes)
	float m_aFlagPos[2][2];    // [team][x/y]
	int8_t m_aFlagCarrier[2];  // -1 if at stand, else player id
	uint8_t m_aFlagAtStand[2]; // Is flag at stand?
	uint8_t m_Padding[12];
	
	// Local tile grid - 29x21 = 609 bytes (matches camera view)
	// Row-major order, centered on player
	// Index = (y + 10) * 29 + (x + 14) where x,y are relative to player
	uint8_t m_aTileGrid[TILE_GRID_SIZE];
	
	// Padding to align to nice size
	uint8_t m_Reserved[31];
};
#pragma pack(pop)

// Total: 8+36+540+640+192+(16+4+12)+609+31 = 2088 bytes
static_assert(sizeof(SStructuredObservation) == 2088, "SStructuredObservation must be 2088 bytes");

} // namespace rl

#endif // GAME_CLIENT_RL_OBSERVATION_H
