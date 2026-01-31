/* Teeworlds RL - Shared Protocol Definitions
 *
 * This header defines structures used for communication between
 * Python RL training code and either:
 * - DDNet client (client-side bots)
 * - DDNet server (server-side bots)
 *
 * Portable: Can be adapted to other Teeworlds-based servers (FNG, iCTF, etc.)
 */
#ifndef GAME_RL_RL_PROTOCOL_H
#define GAME_RL_RL_PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace rl {

// ============================================================================
// Configuration Constants
// ============================================================================

constexpr int MAX_RL_BOTS = 128;

// Maximum number of other players to track per bot
constexpr int MAX_OTHER_PLAYERS = 15;
// Maximum number of projectiles to track
constexpr int MAX_PROJECTILES = 32;
// Maximum number of pickups to track
constexpr int MAX_PICKUPS = 16;

// Local tile grid size (1.5x camera view for better lookahead)
constexpr int TILE_GRID_WIDTH = 43;   // 21 left + 1 center + 21 right
constexpr int TILE_GRID_HEIGHT = 31;  // 15 up + 1 center + 15 down
constexpr int TILE_GRID_SIZE = TILE_GRID_WIDTH * TILE_GRID_HEIGHT; // 1333

// Raycast configuration
constexpr int NUM_RAYCASTS = 16;      // Rays cast in 16 directions (every 22.5 degrees)
constexpr float RAYCAST_MAX_DIST = 640.0f;  // 20 tiles * 32 pixels

// ============================================================================
// Tile Types (for observation)
// ============================================================================

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

// ============================================================================
// Player Observation (per-bot view of a player)
// ============================================================================

#pragma pack(push, 1)
struct SPlayerObs
{
	float m_PosX;
	float m_PosY;
	float m_VelX;
	float m_VelY;

	int8_t m_Direction;
	uint8_t m_Jumped;
	uint8_t m_HookState;
	uint8_t m_Weapon;

	float m_HookPosX;
	float m_HookPosY;
	int8_t m_HookedPlayer;

	uint8_t m_Active;
	uint8_t m_IsFrozen;
	uint8_t m_IsLocal;

	uint8_t m_Health;
	uint8_t m_Armor;

	uint8_t m_Padding[2];
};
#pragma pack(pop)

static_assert(sizeof(SPlayerObs) == 36, "SPlayerObs must be 36 bytes");

// ============================================================================
// Projectile Observation
// ============================================================================

#pragma pack(push, 1)
struct SProjectileObs
{
	float m_PosX;
	float m_PosY;
	float m_VelX;
	float m_VelY;
	uint8_t m_Type;
	uint8_t m_Active;
	uint8_t m_Padding[2];
};
#pragma pack(pop)

static_assert(sizeof(SProjectileObs) == 20, "SProjectileObs must be 20 bytes");

// ============================================================================
// Pickup Observation
// ============================================================================

#pragma pack(push, 1)
struct SPickupObs
{
	float m_PosX;
	float m_PosY;
	uint8_t m_Type;
	uint8_t m_Active;
	uint8_t m_Padding[2];
};
#pragma pack(pop)

static_assert(sizeof(SPickupObs) == 12, "SPickupObs must be 12 bytes");

// ============================================================================
// Full Structured Observation (per bot)
// ============================================================================

#pragma pack(push, 1)
struct SStructuredObservation
{
	// Game state (8 bytes)
	uint32_t m_Tick;
	uint8_t m_GameFlags;
	uint8_t m_GameState;
	uint8_t m_NumActivePlayers;
	uint8_t m_NumProjectiles;

	// This bot's player state (36 bytes)
	SPlayerObs m_LocalPlayer;

	// Other players (36 * 15 = 540 bytes)
	SPlayerObs m_aOtherPlayers[MAX_OTHER_PLAYERS];

	// Projectiles (20 * 32 = 640 bytes)
	SProjectileObs m_aProjectiles[MAX_PROJECTILES];

	// Pickups (12 * 16 = 192 bytes)
	SPickupObs m_aPickups[MAX_PICKUPS];

	// Flags for CTF (24 bytes)
	float m_aFlagPos[2][2];
	int8_t m_aFlagCarrier[2];
	uint8_t m_aFlagAtStand[2];
	uint8_t m_FlagPadding[12];

	// Local tile grid (1333 bytes = 43x31)
	uint8_t m_aTileGrid[TILE_GRID_SIZE];

	// Raycast observations (192 bytes)
	float m_aRaycastSolid[NUM_RAYCASTS];
	float m_aRaycastDeath[NUM_RAYCASTS];
	float m_aRaycastHookable[NUM_RAYCASTS];

	// Extra player state (4 bytes)
	uint8_t m_IsGrounded;
	uint8_t m_JumpsRemaining;
	uint8_t m_CanHook;
	uint8_t m_Reserved2;

	// Checkpoint navigation info (16 bytes)
	float m_NextCheckpointX;   // World X position of next checkpoint
	float m_NextCheckpointY;   // World Y position of next checkpoint
	int32_t m_CheckpointIndex; // Current checkpoint index (0-based, -1 if none reached)
	int32_t m_TotalCheckpoints; // Total number of checkpoints in map

	// Padding for alignment (27 bytes to reach 3020 total)
	uint8_t m_Reserved[27];
};
#pragma pack(pop)

static_assert(sizeof(SStructuredObservation) == 3020, "SStructuredObservation must be 3020 bytes");

// ============================================================================
// Action (from Python to server/client)
// ============================================================================

#pragma pack(push, 1)
struct SAction
{
	int32_t m_Direction;  // -1, 0, 1
	int32_t m_Jump;       // 0 or 1
	int32_t m_Hook;       // 0 or 1
	int32_t m_Fire;       // 0 or 1
	int32_t m_AimX;       // Target X
	int32_t m_AimY;       // Target Y
	int32_t m_Weapon;     // Desired weapon
	int32_t m_Reset;      // Request episode reset
};
#pragma pack(pop)

static_assert(sizeof(SAction) == 32, "SAction must be 32 bytes");

// ============================================================================
// Server-Side Batch Shared Memory Layout
// ============================================================================

#pragma pack(push, 1)
struct SServerRLHeader
{
	// Synchronization
	int32_t m_ServerReady;    // Server has written observations
	int32_t m_PythonReady;    // Python has written actions
	int32_t m_Shutdown;       // Shutdown signal
	int32_t m_Paused;         // Python requests simulation pause (e.g., during PPO update)

	// Configuration
	int32_t m_NumBots;        // Number of active bots
	int32_t m_ServerTick;     // Current server tick
	int32_t m_Padding2[2];
};
#pragma pack(pop)

static_assert(sizeof(SServerRLHeader) == 32, "SServerRLHeader must be 32 bytes");

// Per-bot data in shared memory
#pragma pack(push, 1)
struct SBotData
{
	SAction m_Action;                    // 32 bytes - Python → Server
	SStructuredObservation m_Observation; // 3020 bytes - Server → Python
	float m_Reward;                      // 4 bytes
	int32_t m_Done;                      // 4 bytes
	int32_t m_Truncated;                 // 4 bytes
	int32_t m_BotClientId;               // 4 bytes - Which client ID this bot uses
	uint8_t m_Padding[4];                // Alignment to 3072 bytes
};
#pragma pack(pop)

static_assert(sizeof(SBotData) == 3072, "SBotData must be 3072 bytes (32 + 3020 + 16 + 4)");

// Full shared memory layout
// Total size: 32 + 64 * 3072 = 196,640 bytes (~192 KB)
#pragma pack(push, 1)
struct SServerRLSharedMemory
{
	SServerRLHeader m_Header;
	SBotData m_aBots[MAX_RL_BOTS];
};
#pragma pack(pop)

// ============================================================================
// Helper Functions
// ============================================================================

// Name for shm_open (without /dev/shm prefix)
// ServerIndex allows multiple servers to run in parallel
inline void GetServerRLShmName(char *pBuf, size_t BufSize, int ServerIndex = 0)
{
	if(ServerIndex == 0)
		snprintf(pBuf, BufSize, "/ddnet_rl_server");
	else
		snprintf(pBuf, BufSize, "/ddnet_rl_server_%d", ServerIndex);
}

// Full path for Python (with /dev/shm prefix)
inline void GetServerRLShmPath(char *pBuf, size_t BufSize, int ServerIndex = 0)
{
	if(ServerIndex == 0)
		snprintf(pBuf, BufSize, "/dev/shm/ddnet_rl_server");
	else
		snprintf(pBuf, BufSize, "/dev/shm/ddnet_rl_server_%d", ServerIndex);
}

inline size_t GetServerRLShmSize()
{
	return sizeof(SServerRLSharedMemory);
}

} // namespace rl

#endif // GAME_RL_RL_PROTOCOL_H
