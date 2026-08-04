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
#include <bitset>
#include <memory>
#include <string>
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

	// Initialize with N bots (ServerIndex for multi-server setups)
	bool Init(int NumBots, int ServerIndex = 0);
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

	// Reward calculation
	float CalculateReward(int BotIndex, const SStructuredObservation *pObs, bool IsDecisionTick);
	void ResetRewardState(int BotIndex, int SpawnCheckpoint = -1);

	// Normalised path progress in [0, 1] for a bot, advancing its tracked
	// waypoint index. The only progress coordinate that is well-defined on every
	// race map — see roadmap.md §1.7.
	float GetNormalizedProgress(int BotIndex, vec2 Pos);

	// Egocentric path lookahead written into the observation.
	void FillLookahead(int BotIndex, vec2 Pos, SStructuredObservation *pObs);

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

	// Per-bot reward tracking state.
	//
	// The freeze/stuck/checkpoint-distance bookkeeping that used to live here is
	// gone along with the reward terms that consumed it (roadmap.md Part 1b).
	// What remains is the tracked waypoint index and the previous potential.
	struct SBotRewardState
	{
		float m_LastPosX;
		float m_LastPosY;
		int m_LastHookState;
		bool m_WasDead;
		int m_HighestTimeCp;  // Highest time checkpoint reached. Observation only —
		                      // NOT used by the reward. Removed in the Phase 1 protocol batch.

		// Path progress tracking
		int m_WaypointIndex;  // Tracked waypoint index; only moves within a
		                      // physically plausible window (see FindClosestWaypoint)
		float m_LastPhi;      // Previous normalised progress, for potential-based shaping

		// Exploration tracking - coarse grid (8x8 tiles per region)
		// 64x64 regions = 4096 bits = 512 bytes per bot
		static constexpr int EXPLORATION_GRID_SIZE = 64;
		std::bitset<EXPLORATION_GRID_SIZE * EXPLORATION_GRID_SIZE> m_VisitedRegions;
		int m_NewRegionsThisEpisode;
	};
	SBotRewardState m_aBotState[MAX_RL_BOTS];

	// Map checkpoint data
	std::vector<SCheckpoint> m_vCheckpoints;
	vec2 m_SpawnPos;
	vec2 m_FinishPos;
	bool m_MapAnalyzed;

	// ------------------------------------------------------------------
	// Reward constants (roadmap.md Part 1b)
	//
	//   R = -TIME_PENALTY  +  SHAPING_K * (GAMMA * Phi' - Phi)  +  FINISH_BONUS
	//
	// Phi is normalised path progress in [0, 1], so the reward scale no longer
	// depends on map length (the old Delta/100 gave a long map 10x the total
	// reward of a short one — another cross-map inconsistency).
	//
	// Scale derivation. Forward motion must stay net-positive, or the agent
	// prefers to die immediately and stop paying the time penalty:
	//
	//   progress rate at v units/s = SHAPING_K * v / PathLength  per second
	//   time cost                  = TIME_PENALTY * 50           per second
	//
	// A mid-training agent runs ~150 units/s and the ghost runs ~600. On
	// Tutorial (112k units) with SHAPING_K = 100 and TIME_PENALTY = 0.001:
	//   slow agent: +0.134/s progress vs -0.05/s time  -> net positive
	//   ghost pace: +0.536/s progress vs -0.05/s time  -> ~4x better
	// so speed is rewarded without ever making death attractive. The old value
	// (-0.0005/tick against ~1.5/s of progress reward) was a 60:1 ratio, which
	// is why the agent ambled at 25% of ghost pace.
	static constexpr float TIME_PENALTY = 0.001f;  // per tick
	static constexpr float SHAPING_K = 100.0f;     // reward for the full path
	static constexpr float FINISH_BONUS = 100.0f;

	// Discount used INSIDE the shaping term. Deliberately 1.0, not the PPO gamma.
	//
	// With gamma < 1 the shaping sum expands to
	//     K*gamma*Phi_T  -  K*(1-gamma)*sum_t(Phi_t)
	// and that second term is a per-step leak of -K*(1-gamma)*Phi. Over an
	// 800-decision episode at gamma=0.99 the leak is -20.0 against a useful
	// signal of +4.95 — a drift/signal ratio of (1-gamma)*T = 8x that GROWS the
	// further the agent gets, so episodic return moves the wrong way while the
	// policy genuinely improves.
	//
	// The strict Ng/Harada/Russell invariance result wants gamma here to match
	// the agent's discount. Using 1.0 gives up that guarantee, but in exchange
	// the shaping telescopes cleanly to K*(Phi_T - Phi_0): the return is
	// monotone in progress, and the critic no longer has to represent and cancel
	// a state-dependent term 16x the size of the time penalty. This is the
	// standard practical choice.
	static constexpr float SHAPING_GAMMA = 1.0f;

	// Waypoint-based reward system
	struct SWaypoint
	{
		float m_X;
		float m_Y;
		float m_CumulativeDist; // Distance from start to this waypoint along path
		int m_SegmentId;        // Path segment; increments across a teleport/discontinuity
	};
	std::vector<SWaypoint> m_vWaypoints;

	// Temporal-coherence window for waypoint matching. A merely spatial search
	// matches the wrong strand wherever the path doubles back on itself — on
	// Tutorial the path passes within ~100 units of a point ~1000 units later,
	// only ~17 waypoints apart, well inside the old +/-100 window. Sized from
	// physics instead: a tee tops out near 100 units/tick.
	static constexpr float MAX_TEE_SPEED = 100.0f;
	static constexpr int MAX_RETREAT = 2;
	static constexpr float LOST_DIST = 512.0f;  // beyond this, scan forward to re-acquire
	int m_MaxAdvance;  // derived from waypoint spacing at load time
	void SegmentPathAtDiscontinuities();
	void UpdateMatchWindow();
	float m_TotalPathLength;
	bool m_WaypointsLoaded;
	bool LoadWaypoints(const std::string &Path);
	int FindClosestWaypoint(vec2 Pos, int SearchStart, int SearchRadius = 50) const;
	float GetPathProgress(vec2 Pos, int WaypointIdx) const;

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

	// Sync flag: only signal observations when we actually processed new actions
	bool m_ShouldSignalObservations;
	bool m_InitialObservationsSent;  // Track if initial observations have been sent

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
