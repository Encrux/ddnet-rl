/* DDNet Server RL - Bot Manager
 *
 * Manages server-side bot players for RL training.
 * Creates bot players, injects inputs from Python, extracts observations.
 *
 * Portable: Can be adapted to other Teeworlds-based servers.
 */
#ifndef GAME_SERVER_RL_BOTMANAGER_H
#define GAME_SERVER_RL_BOTMANAGER_H

#include <game/rl/rl_protocol.h>
#include <base/vmath.h>
#include <generated/protocol.h>
#include <memory>
#include <vector>
#include <map>

class CGameContext;
class CPlayer;
class CCharacter;
class CCollision;

namespace rl {

class CServerShm;

class CBotManager
{
public:
	CBotManager(CGameContext *pGameContext);
	~CBotManager();

	// Initialize with N bots
	bool Init(int NumBots);
	void Shutdown();

	// Called each server tick
	void Tick();

	// Check if enabled
	bool IsEnabled() const { return m_Enabled; }
	int NumBots() const { return m_NumBots; }

	// Frame skip configuration
	void SetFrameSkip(int FrameSkip);
	int GetFrameSkip() const { return m_FrameSkip; }

	// Pause support (for PPO updates)
	void SetAllowPause(bool Allow) { m_AllowPause = Allow; }
	bool GetAllowPause() const { return m_AllowPause; }
	bool IsPaused() const;  // Returns true if Python requested pause AND allow_pause is enabled

	// Get bot client ID
	int GetBotClientId(int BotIndex) const;

	// Force respawn all bots (e.g., on map change)
	void RespawnAllBots();

	// Extract observations from all bots (call after world tick)
	void ExtractObservations();

private:
	// Create bot players
	bool CreateBotPlayers();

	// Input injection
	void InjectBotInputs();
	void ConvertActionToInput(const SAction *pAction, CNetObj_PlayerInput *pInput);

	// Observation extraction helpers
	void ExtractPlayerObs(CCharacter *pChar, SPlayerObs *pObs, bool IsLocal);
	void ExtractTileGrid(CCharacter *pChar, uint8_t *pGrid);
	void ExtractRaycasts(CCharacter *pChar, SStructuredObservation *pObs);
	float CastRay(vec2 Start, vec2 Dir, float MaxDist, int TileMask);

	// Reward calculation
	float CalculateReward(int BotIndex, const SStructuredObservation *pObs);
	void ResetRewardState(int BotIndex);

	// Spawn position for respawn
	vec2 GetSpawnPos();

	CGameContext *m_pGameContext;
	CCollision *m_pCollision;
	std::unique_ptr<CServerShm> m_pShm;

	bool m_Enabled;
	bool m_PlayersCreated;  // True once actual players have been created
	int m_NumBots;
	int m_aClientIds[MAX_RL_BOTS];  // Client IDs assigned to bots

	// Checkpoint data extracted from map
	struct SCheckpoint
	{
		vec2 m_Pos;
		int m_Index;
	};

	// Per-bot reward tracking state
	struct SBotRewardState
	{
		float m_LastPosX;
		float m_LastPosY;
		float m_LastDistToCheckpoint;
		int m_CurrentCheckpointIndex;
		int m_LastHookState;
		int m_StuckCounter;
		bool m_WasFrozen;
		bool m_WasDead;
	};
	SBotRewardState m_aBotState[MAX_RL_BOTS];

	// Map checkpoint data
	std::vector<SCheckpoint> m_vCheckpoints;
	vec2 m_SpawnPos;
	vec2 m_FinishPos;
	bool m_MapAnalyzed;

	// Analyze map for checkpoints (called once on first tick)
	void AnalyzeMap();
	vec2 GetNextCheckpointPos(int CurrentIndex) const;
	int GetCheckpointAtPos(vec2 Pos, float Radius = 48.0f) const;

	// Input storage
	CNetObj_PlayerInput m_aBotInputs[MAX_RL_BOTS];

	// Tick tracking
	int m_LastTick;

	// Frame skip (only exchange with Python every N ticks)
	int m_FrameSkip;
	int m_FrameSkipCounter;
	float m_aAccumulatedRewards[MAX_RL_BOTS];

	// Pause support
	bool m_AllowPause;
	bool m_WasPaused;  // Track pause state transitions for latency reset

	// Diagnostic stats
	int m_TotalTicks;
	int m_MissedActions;
	int m_ConsecutiveMisses;
	int m_MaxConsecutiveMisses;
	int64_t m_LastSuccessTime;
	int64_t m_MaxLatencyUs;

public:
	// Get diagnostic stats
	void GetStats(int *pTotalTicks, int *pMissedActions, int *pMaxConsecutive, int64_t *pMaxLatencyUs) const;
};

} // namespace rl

#endif // GAME_SERVER_RL_BOTMANAGER_H
