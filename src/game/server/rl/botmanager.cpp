/* DDNet Server RL - Bot Manager Implementation */
#include "botmanager.h"
#include "server_shm.h"

#include <game/server/gamecontext.h>
#include <game/server/gamecontroller.h>
#include <game/server/player.h>
#include <game/server/entities/character.h>
#include <game/collision.h>
#include <game/mapitems.h>

#include <engine/shared/config.h>
#include <engine/server.h>
#include <generated/protocol.h>

#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace rl {

CBotManager::CBotManager(CGameContext *pGameContext) :
	m_pGameContext(pGameContext),
	m_pCollision(nullptr),
	m_Enabled(false),
	m_PlayersCreated(false),
	m_NumBots(0),
	m_SpawnPos(0, 0),
	m_FinishPos(0, 0),
	m_MapAnalyzed(false),
	m_SnapshotsCaptured(0),
	m_MaxAdvance(8),
	m_TotalPathLength(0.0f),
	m_WaypointsLoaded(false),
	m_LastTick(0),
	m_FrameSkip(1),
	m_FrameSkipCounter(0),
	m_AllowPause(true),
	m_WasPaused(false),
	m_ShouldSignalObservations(false),
	m_InitialObservationsSent(false),
	m_TotalTicks(0),
	m_MissedActions(0),
	m_ConsecutiveMisses(0),
	m_MaxConsecutiveMisses(0),
	m_LastSuccessTime(0),
	m_MaxLatencyUs(0)
{
	mem_zero(m_aClientIds, sizeof(m_aClientIds));
	mem_zero(m_aBotInputs, sizeof(m_aBotInputs));
	mem_zero(m_aAccumulatedRewards, sizeof(m_aAccumulatedRewards));

	// Initialize bot state (can't use mem_zero due to std::bitset)
	for(int i = 0; i < MAX_RL_BOTS; i++)
	{
		m_aBotState[i] = SBotRewardState{};
	}
}

CBotManager::~CBotManager()
{
	Shutdown();
}

bool CBotManager::Init(int NumBots, int ServerIndex)
{
	if(m_Enabled)
	{
		dbg_msg("rl/bots", "Bot manager already enabled");
		return false;
	}

	if(NumBots <= 0 || NumBots > MAX_RL_BOTS)
	{
		dbg_msg("rl/bots", "Invalid number of bots: %d (max %d)", NumBots, MAX_RL_BOTS);
		return false;
	}

	m_NumBots = NumBots;
	m_pCollision = m_pGameContext->Collision();

	// Initialize shared memory
	m_pShm = std::make_unique<CServerShm>();
	if(!m_pShm->Init(NumBots, ServerIndex))
	{
		dbg_msg("rl/bots", "Failed to initialize shared memory");
		m_pShm.reset();
		return false;
	}

	// Create bot players
	if(!CreateBotPlayers())
	{
		dbg_msg("rl/bots", "Failed to create bot players");
		m_pShm->Shutdown();
		m_pShm.reset();
		return false;
	}

	m_Enabled = true;
	m_LastTick = m_pGameContext->Server()->Tick();

	dbg_msg("rl/bots", "Bot manager initialized with %d bots", NumBots);
	return true;
}

void CBotManager::Shutdown()
{
	if(!m_Enabled)
		return;

	// Remove bot players
	for(int i = 0; i < m_NumBots; i++)
	{
		int ClientId = m_aClientIds[i];
		if(ClientId >= 0 && m_pGameContext->m_apPlayers[ClientId])
		{
			delete m_pGameContext->m_apPlayers[ClientId];
			m_pGameContext->m_apPlayers[ClientId] = nullptr;
		}
	}

	// Shutdown shared memory
	if(m_pShm)
	{
		m_pShm->Shutdown();
		m_pShm.reset();
	}

	m_Enabled = false;
	m_NumBots = 0;
	mem_zero(m_aClientIds, sizeof(m_aClientIds));
	m_vCheckpoints.clear();
	m_MapAnalyzed = false;

	dbg_msg("rl/bots", "Bot manager shutdown");
}

bool CBotManager::CreateBotPlayers()
{
	// Use SERVER_MAX_CLIENTS (64) not MAX_CLIENTS (128) because the network layer
	// only supports 64 clients. Bot IDs must be in range 0-63.
	dbg_msg("rl/bots", "CreateBotPlayers: Creating %d bots, SERVER_MAX_CLIENTS=%d", m_NumBots, SERVER_MAX_CLIENTS);

	// Check if game controller is ready
	if(!m_pGameContext->m_pController)
	{
		dbg_msg("rl/bots", "Game controller not ready, using virtual bots only");
	}

	// Find available client IDs starting from the end (real players use lower IDs)
	// Use SERVER_MAX_CLIENTS to stay within valid network client range
	int NextClientId = SERVER_MAX_CLIENTS - 1;

	for(int i = 0; i < m_NumBots; i++)
	{
		// Find an unused client ID
		while(NextClientId >= 0 && m_pGameContext->m_apPlayers[NextClientId] != nullptr)
		{
			NextClientId--;
		}

		if(NextClientId < 0)
		{
			dbg_msg("rl/bots", "No more client IDs available for bots");
			return false;
		}

		m_aClientIds[i] = NextClientId;

		// Try to create actual player if controller is ready
		if(m_pGameContext->m_pController)
		{
			// Initialize bot client state with a name BEFORE creating player
			char aBotName[MAX_NAME_LENGTH];
			str_format(aBotName, sizeof(aBotName), "RL Bot %03d", i);
			m_pGameContext->Server()->InitBotClient(NextClientId, aBotName);

			// Use CreatePlayer which handles the pool allocation and UniqueClientId
			// TEAM_GAME = 0 (TEAM_RED)
			CPlayer *pPlayer = m_pGameContext->CreatePlayer(NextClientId, 0, false, -1);

			if(pPlayer)
			{
				pPlayer->m_IsBot = true;

				// Spawn character
				pPlayer->Respawn();

				dbg_msg("rl/bots", "Created bot %d with client ID %d (with player)", i, NextClientId);
			}
			else
			{
				dbg_msg("rl/bots", "Failed to create player for bot %d", i);
			}
		}
		else
		{
			dbg_msg("rl/bots", "Allocated virtual bot %d with client ID %d", i, NextClientId);
		}

		// Initialize reward state
		ResetRewardState(i, GetSpawnPos());

		// Store bot client ID in shared memory (contiguous layout)
		int32_t *pBotClientId = m_pShm->GetBotClientId(i);
		if(pBotClientId)
		{
			*pBotClientId = NextClientId;
		}

		NextClientId--;
	}

	dbg_msg("rl/bots", "Created %d bots (client IDs %d-%d)",
		m_NumBots, m_aClientIds[m_NumBots - 1], m_aClientIds[0]);

	return true;
}

int CBotManager::GetBotClientId(int BotIndex) const
{
	if(BotIndex < 0 || BotIndex >= m_NumBots)
		return -1;
	return m_aClientIds[BotIndex];
}

void CBotManager::Tick()
{
	if(!m_Enabled || !m_pShm)
		return;

	// Skip tick processing when paused (e.g., during PPO updates)
	// This prevents counting misses and incrementing tick counters during pause
	if(IsPaused())
	{
		m_WasPaused = true;
		return;
	}

	// Reset latency timer after unpausing to avoid counting pause time
	if(m_WasPaused)
	{
		m_LastSuccessTime = 0;  // Will be set on next successful exchange
		m_WasPaused = false;
	}

	// Deferred player creation: create actual players once controller is ready
	if(!m_PlayersCreated && m_pGameContext->m_pController)
	{
		dbg_msg("rl/bots", "Game controller ready, creating bot players...");
		for(int i = 0; i < m_NumBots; i++)
		{
			int ClientId = m_aClientIds[i];

			// Create player if not already exists
			if(!m_pGameContext->m_apPlayers[ClientId])
			{
				// Initialize bot client state with a name BEFORE creating player
				char aBotName[MAX_NAME_LENGTH];
				str_format(aBotName, sizeof(aBotName), "RL Bot %03d", i);
				m_pGameContext->Server()->InitBotClient(ClientId, aBotName);

				CPlayer *pPlayer = m_pGameContext->CreatePlayer(ClientId, 0, false, -1);
				if(pPlayer)
				{
					pPlayer->m_IsBot = true;

					// Use ForceSpawn instead of Respawn because bot players
					// don't have network clients, so Server()->ClientIngame() is false
					// and normal TryRespawn() would never be called
					vec2 SpawnPos = GetSpawnPos();
					pPlayer->ForceSpawn(SpawnPos);

					dbg_msg("rl/bots", "Created player for bot %d (client %d) at (%.0f, %.0f)",
						i, ClientId, SpawnPos.x, SpawnPos.y);
				}
			}
		}
		m_PlayersCreated = true;
	}

	// Check for shutdown request
	if(m_pShm->IsShutdownRequested())
	{
		dbg_msg("rl/bots", "Shutdown requested by Python");
		Shutdown();
		return;
	}

	m_TotalTicks++;

	// Frame skip: only exchange with Python on decision ticks
	// On non-decision ticks, previous inputs are automatically reused
	m_FrameSkipCounter++;
	if(m_FrameSkipCounter < m_FrameSkip)
	{
		// Not a decision tick - skip Python exchange, use previous inputs
		return;
	}
	m_FrameSkipCounter = 0; // Reset counter for next cycle

	// Decision tick: check if Python has actions ready (non-blocking)
	int64_t Now = time_get();
	if(!m_pShm->AreActionsReady())
	{
		// No actions ready - use previous inputs
		// Don't block - the server needs to process network events
		m_MissedActions++;
		m_ConsecutiveMisses++;
		if(m_ConsecutiveMisses > m_MaxConsecutiveMisses)
			m_MaxConsecutiveMisses = m_ConsecutiveMisses;
		// Mark that we should NOT signal observations this tick
		m_ShouldSignalObservations = false;
		return;
	}

	// Track latency since last successful exchange
	if(m_LastSuccessTime > 0)
	{
		int64_t LatencyUs = (Now - m_LastSuccessTime) * 1000000 / time_freq();
		if(LatencyUs > m_MaxLatencyUs)
			m_MaxLatencyUs = LatencyUs;
	}
	m_LastSuccessTime = Now;

	m_ConsecutiveMisses = 0; // Reset on success

	// Inject inputs
	InjectBotInputs();

	// Clear action ready flag (but not server_ready - we'll set that after observations)
	SServerRLHeader *pHeader = m_pShm->GetHeader();
	if(pHeader)
		pHeader->m_PythonReady = 0;

	// Mark that we should signal observations after ExtractObservations
	m_ShouldSignalObservations = true;
}

void CBotManager::InjectBotInputs()
{
	for(int i = 0; i < m_NumBots; i++)
	{
		int ClientId = m_aClientIds[i];

		// Get action from shared memory (contiguous layout)
		SAction *pAction = m_pShm->GetAction(i);
		if(!pAction)
			continue;

		// Check if we have a real player for this bot
		CPlayer *pPlayer = m_pGameContext->m_apPlayers[ClientId];

		// Handle reset request
		if(pAction->m_Reset)
		{
			if(pPlayer && pPlayer->GetCharacter())
			{
				pPlayer->KillCharacter(WEAPON_GAME, false);
			}
			if(pPlayer)
			{
				// Spawn first — CSaveTee::Load needs a live character to write into.
				vec2 SpawnPos = GetSpawnPos();
				pPlayer->ForceSpawn(SpawnPos);

				CCharacter *pNewChar = pPlayer->GetCharacter();
				if(pAction->m_RestoreSlot >= 0 && RestoreSnapshot(pAction->m_RestoreSlot, pNewChar))
					SpawnPos = pNewChar->GetPos();  // restore moved us; re-seed from there

				ResetRewardState(i, SpawnPos);
			}
			else
			{
				ResetRewardState(i, GetSpawnPos());
			}
			pAction->m_Reset = 0;
			pAction->m_RestoreSlot = -1;
		}

		// Convert action to input
		ConvertActionToInput(pAction, &m_aBotInputs[i]);

		// Only inject if we have a real player
		if(pPlayer)
		{
			// Store as last input
			m_pGameContext->m_aLastPlayerInput[ClientId] = m_aBotInputs[i];
			m_pGameContext->m_aPlayerHasInput[ClientId] = true;

			// Inject input
			CCharacter *pChar = pPlayer->GetCharacter();
			if(pChar)
			{
				pPlayer->OnDirectInput(&m_aBotInputs[i]);
				pPlayer->OnPredictedInput(&m_aBotInputs[i]);
			}
		}
	}
}

void CBotManager::ConvertActionToInput(const SAction *pAction, CNetObj_PlayerInput *pInput)
{
	mem_zero(pInput, sizeof(CNetObj_PlayerInput));

	pInput->m_Direction = pAction->m_Direction;
	pInput->m_Jump = pAction->m_Jump;
	pInput->m_Hook = pAction->m_Hook;
	pInput->m_Fire = pAction->m_Fire ? 1 : 0;
	pInput->m_TargetX = pAction->m_AimX;
	pInput->m_TargetY = pAction->m_AimY;
	pInput->m_WantedWeapon = pAction->m_Weapon + 1; // 1-indexed
	pInput->m_PlayerFlags = 0;
}

void CBotManager::ExtractObservations()
{
	if(!m_Enabled || !m_pShm)
		return;

	// Skip during pause (no observations to extract, physics frozen)
	if(IsPaused())
		return;

	// Analyze map for checkpoints on first call
	if(!m_MapAnalyzed && m_pCollision)
	{
		AnalyzeMap();
	}

	int CurrentTick = m_pGameContext->Server()->Tick();

	// Check if this is a decision tick (when we'll signal Python)
	// m_FrameSkipCounter was already incremented and potentially reset in Tick()
	// So if it's 0, we just reset it, meaning this is a decision tick
	bool IsDecisionTick = (m_FrameSkipCounter == 0);

	for(int i = 0; i < m_NumBots; i++)
	{
		int ClientId = m_aClientIds[i];

		// Get pointers from contiguous layout
		SStructuredObservation *pObs = m_pShm->GetObservation(i);
		float *pReward = m_pShm->GetReward(i);
		int32_t *pDone = m_pShm->GetDone(i);

		if(!pObs || !pReward || !pDone)
			continue;

		// Check if we have a real player with character
		CPlayer *pPlayer = m_pGameContext->m_apPlayers[ClientId];
		CCharacter *pChar = pPlayer ? pPlayer->GetCharacter() : nullptr;

		float TickReward = 0.0f;
		bool IsDone = false;

		if(pChar && pChar->IsAlive())
		{
			// On decision ticks, extract full observation
			// On non-decision ticks, we still need to calculate reward for accumulation
			if(IsDecisionTick)
			{
				mem_zero(pObs, sizeof(SStructuredObservation));

				pObs->m_Tick = CurrentTick;

				// Extract local player state
				ExtractPlayerObs(pChar, &pObs->m_LocalPlayer, true);

				// Extract tile grid (full 43x31 for CNN)
				ExtractTileGrid(pChar, pObs->m_aTileGrid);

				// Note: Raycasts and other players removed for solo race CNN training

				// Extra state
				vec2 Pos = pChar->GetPos();
				int TileX = (int)(Pos.x / 32.0f);
				int TileY = (int)((Pos.y + 16.0f) / 32.0f); // Check below feet

				pObs->m_IsGrounded = 0;
				if(m_pCollision && TileX >= 0 && TileX < m_pCollision->GetWidth() &&
				   TileY >= 0 && TileY < m_pCollision->GetHeight())
				{
					int MapIndex = TileY * m_pCollision->GetWidth() + TileX;
					int TileIdx = m_pCollision->GetTileIndex(MapIndex);
					if(TileIdx == TILE_SOLID)
						pObs->m_IsGrounded = 1;
				}

				// Jumps remaining
				int Jumped = pChar->Core()->m_Jumped;
				if(Jumped == 0)
					pObs->m_JumpsRemaining = 2;
				else if(Jumped == 1)
					pObs->m_JumpsRemaining = 1;
				else
					pObs->m_JumpsRemaining = 0;

				pObs->m_CanHook = (pChar->Core()->m_HookState == HOOK_IDLE) ? 1 : 0;

				// Navigation: normalised progress + geometric path lookahead.
				// Time checkpoints are gone entirely — see rl_protocol.h.
				SBotRewardState &State = m_aBotState[i];
				State.m_HighestTimeCp = std::max(State.m_HighestTimeCp, pChar->m_LastTimeCp);

				// CalculateReward advances the tracked waypoint index, so fill the
				// lookahead after it to stay consistent with this tick's match.
				TickReward = CalculateReward(i, pObs, true);

				pObs->m_Progress = State.m_LastPhi;
				pObs->m_MaxProgress = State.m_MaxPhi;
				FillLookahead(i, Pos, pObs);
				CaptureSnapshot(i, pChar, State.m_LastPhi);
			}
			else
			{
				// Non-decision tick: only the per-tick time penalty accrues. The
				// shaping term must fire once per decision, not once per tick.
				SStructuredObservation TempObs;
				mem_zero(&TempObs, sizeof(TempObs));
				ExtractPlayerObs(pChar, &TempObs.m_LocalPlayer, true);

				TickReward = CalculateReward(i, &TempObs, false);
			}
		}
		else
		{
			// Character dead or not spawned. No explicit death penalty: with a
			// positive-in-expectation forward reward, terminating forfeits the
			// remaining shaping and the finish bonus, which is the cost. Adding a
			// constant here would be a fourth term with an arbitrary scale.
			IsDone = true;
			dbg_msg("rl/done", "Bot %d: dead/not spawned", i);
		}

		// Accumulate reward
		m_aAccumulatedRewards[i] += TickReward;

		// On decision tick, write accumulated reward to shared memory and reset
		if(IsDecisionTick)
		{
			*pReward = m_aAccumulatedRewards[i];
			*pDone = IsDone ? 1 : 0;
			m_aAccumulatedRewards[i] = 0.0f; // Reset for next cycle
		}
	}

	// Signal observations when:
	// 1. First tick (initial observations for Python to start)
	// 2. After successfully processing actions from Python
	bool ShouldSignal = IsDecisionTick && (m_ShouldSignalObservations || !m_InitialObservationsSent);
	if(ShouldSignal)
	{
		// Update header tick
		SServerRLHeader *pHeader = m_pShm->GetHeader();
		if(pHeader)
		{
			pHeader->m_ServerTick = CurrentTick;
		}

		// Signal observations are ready
		m_pShm->SignalObservationsReady();
		m_ShouldSignalObservations = false;  // Reset until next successful action read
		m_InitialObservationsSent = true;
	}

	m_LastTick = CurrentTick;
}

void CBotManager::ExtractPlayerObs(CCharacter *pChar, SPlayerObs *pObs, bool IsLocal)
{
	mem_zero(pObs, sizeof(SPlayerObs));

	if(!pChar)
		return;

	const CCharacterCore *pCore = pChar->Core();
	if(!pCore)
		return;

	pObs->m_Active = 1;
	pObs->m_IsLocal = IsLocal ? 1 : 0;

	pObs->m_PosX = pCore->m_Pos.x;
	pObs->m_PosY = pCore->m_Pos.y;
	pObs->m_VelX = pCore->m_Vel.x;
	pObs->m_VelY = pCore->m_Vel.y;

	pObs->m_Direction = pCore->m_Direction;
	pObs->m_Jumped = pCore->m_Jumped;
	pObs->m_HookState = pCore->m_HookState;
	pObs->m_Weapon = pChar->GetActiveWeapon();

	pObs->m_HookPosX = pCore->m_HookPos.x;
	pObs->m_HookPosY = pCore->m_HookPos.y;
	pObs->m_HookedPlayer = pCore->HookedPlayer();

	pObs->m_Health = pChar->GetHealth();
	pObs->m_Armor = pChar->GetArmor();

	// Freeze state
	pObs->m_IsFrozen = pChar->m_FreezeTime > 0 ? 1 : 0;
}

void CBotManager::ExtractTileGrid(CCharacter *pChar, uint8_t *pGrid)
{
	if(!pChar || !m_pCollision)
	{
		mem_zero(pGrid, TILE_GRID_SIZE);
		return;
	}

	vec2 Pos = pChar->GetPos();
	int PlayerTileX = (int)(Pos.x / 32.0f);
	int PlayerTileY = (int)(Pos.y / 32.0f);

	int MapWidth = m_pCollision->GetWidth();
	int MapHeight = m_pCollision->GetHeight();

	// Center offsets for the tile grid (player is at center)
	const int HalfWidth = TILE_GRID_WIDTH / 2;   // 21 for 43-wide grid
	const int HalfHeight = TILE_GRID_HEIGHT / 2; // 15 for 31-high grid

	// Get teleport layer once outside the loop for efficiency
	const CTeleTile *pTele = m_pCollision->TeleLayer();

	for(int gy = 0; gy < TILE_GRID_HEIGHT; gy++)
	{
		for(int gx = 0; gx < TILE_GRID_WIDTH; gx++)
		{
			int TileX = PlayerTileX + (gx - HalfWidth);
			int TileY = PlayerTileY + (gy - HalfHeight);

			uint8_t TileType = 0;  // TILE_AIR

			if(TileX >= 0 && TileX < MapWidth && TileY >= 0 && TileY < MapHeight)
			{
				int MapIndex = TileY * MapWidth + TileX;
				int TileIdx = m_pCollision->GetTileIndex(MapIndex);

				// Check teleport layer - teleports are on a separate layer
				// If a teleport exists, use its type instead of the underlying tile
				if(pTele && pTele[MapIndex].m_Type != 0)
				{
					TileIdx = pTele[MapIndex].m_Type;
				}

				// Pass raw tile index (clamped to uint8 range)
				// This preserves all tile types for learned embeddings
				TileType = (TileIdx > 255) ? 255 : (uint8_t)TileIdx;
			}
			else
			{
				TileType = 1;  // TILE_SOLID for out of bounds
			}

			pGrid[gy * TILE_GRID_WIDTH + gx] = TileType;
		}
	}
}

// Egocentric offsets to points ahead along the reference path. Where the path
// runs out (near the finish, or past the last waypoint) the remaining slots
// clamp to the final waypoint rather than going zero, so "nothing ahead" reads
// as "the end is over there" instead of "the target is on top of me".
void CBotManager::FillLookahead(int BotIndex, vec2 Pos, SStructuredObservation *pObs)
{
	if(!m_WaypointsLoaded || m_vWaypoints.empty())
	{
		for(int k = 0; k < NUM_LOOKAHEAD; k++)
		{
			pObs->m_aLookaheadX[k] = 0.0f;
			pObs->m_aLookaheadY[k] = 0.0f;
		}
		return;
	}

	const int Base = m_aBotState[BotIndex].m_WaypointIndex;
	const int Last = (int)m_vWaypoints.size() - 1;
	for(int k = 0; k < NUM_LOOKAHEAD; k++)
	{
		int Idx = std::min(Base + LOOKAHEAD_OFFSETS[k], Last);
		pObs->m_aLookaheadX[k] = m_vWaypoints[Idx].m_X - Pos.x;
		pObs->m_aLookaheadY[k] = m_vWaypoints[Idx].m_Y - Pos.y;
	}
}

int CBotManager::ProgressToBin(float Phi)
{
	int Bin = (int)(Phi * NUM_PROGRESS_BINS);
	return std::clamp(Bin, 0, NUM_PROGRESS_BINS - 1);
}

// Latest-wins per bin. A snapshot from the current policy is a more useful
// restore point than a stale one from an earlier, worse policy, and it keeps the
// archive self-refreshing without needing a "better" metric or per-bot timers.
void CBotManager::CaptureSnapshot(int BotIndex, CCharacter *pChar, float Phi)
{
	if(!pChar || !pChar->IsAlive() || m_vSnapshots.empty())
		return;

	SBotRewardState &State = m_aBotState[BotIndex];
	int Bin = ProgressToBin(Phi);
	if(Bin == State.m_LastBin)
		return;  // only on bin transitions, not every decision tick
	State.m_LastBin = Bin;

	// AddPenalty = false: this is not a /save, we do not want the time penalty.
	m_vSnapshots[Bin].Save(pChar, false);
	if(!m_vSnapshotValid[Bin])
		m_SnapshotsCaptured++;
	m_vSnapshotValid[Bin] = 1;
}

bool CBotManager::RestoreSnapshot(int Slot, CCharacter *pChar)
{
	if(!pChar || Slot < 0 || Slot >= (int)m_vSnapshots.size() || !m_vSnapshotValid[Slot])
		return false;
	// Team = nullopt: skip team bookkeeping entirely. Load() never reassigns the
	// client id, so a snapshot taken by any bot restores into any other.
	return m_vSnapshots[Slot].Load(pChar, std::nullopt);
}

float CBotManager::GetNormalizedProgress(int BotIndex, vec2 Pos)
{
	if(!m_WaypointsLoaded || m_TotalPathLength <= 0.0f)
		return 0.0f;

	SBotRewardState &State = m_aBotState[BotIndex];
	State.m_WaypointIndex = FindClosestWaypoint(Pos, State.m_WaypointIndex, m_MaxAdvance);
	return GetPathProgress(Pos, State.m_WaypointIndex) / m_TotalPathLength;
}

// Three terms, and deliberately no more (roadmap.md Part 1b).
//
// The freeze penalty, stuck ramp, off-path penalty, checkpoint bonus and
// velocity-toward-checkpoint term are all gone. Each was a hand-tuned scale with
// no reason to hold on the next map, and each could move the optimal policy. The
// per-tick freeze penalty in particular cost -5.0/s against a maximum progress
// rate of +5.0/s, so one second frozen exactly cancelled one second of
// world-record-pace running — on a game where passing through freeze is routine
// and sometimes mandatory.
//
// The progress term is a POTENTIAL DIFFERENCE, not the old high-water ratchet.
// Ng, Harada & Russell (1999) proved that shaping of the form
// gamma*Phi(s') - Phi(s) leaves the optimal policy unchanged for ANY Phi. That is
// exactly the robustness we need: a mediocre ghost line, a suboptimal pathfinder
// route, or a garbage segment near a teleport can cost learning speed but can
// never teach the wrong behaviour. The ratchet had no such guarantee — being
// non-Markovian it did move the optimum, and it made every tick after a failed
// frontier attempt strictly negative with no way to earn anything back.
float CBotManager::CalculateReward(int BotIndex, const SStructuredObservation *pObs, bool IsDecisionTick)
{
	float Reward = 0.0f;
	SBotRewardState &State = m_aBotState[BotIndex];
	const SPlayerObs &Local = pObs->m_LocalPlayer;

	if(!Local.m_Active)
		return 0.0f;

	vec2 Pos(Local.m_PosX, Local.m_PosY);

	// Time cost accrues every tick — this is the actual race objective.
	Reward -= TIME_PENALTY;

	// Potential-based shaping, applied ONCE PER DECISION.
	//
	// This must not run per tick. Reward is accumulated across the frame skip,
	// and summing (gamma*Phi_{t+1} - Phi_t) over N ticks does not equal
	// gamma^N*Phi_{t+N} - Phi_t, so the telescoping that makes the shaping
	// policy-invariant breaks. The discount here has to be the agent's gamma at
	// the agent's decision rate.
	if(IsDecisionTick && m_WaypointsLoaded)
	{
		float Phi = GetNormalizedProgress(BotIndex, Pos);

		if(m_RatchetShaping)
		{
			// Potential is the HIGH-WATER progress, not the current progress.
			//
			// Plain shaping charges for every unit of ground given back, which
			// makes the first step of any obstacle that needs a run-up
			// immediately and certainly negative. At the 0.455 wall the correct
			// play starts by backing up ~3 tiles: -0.086 reward, about 29
			// decisions' worth of time penalty, paid up front against a payoff
			// that during exploration almost never arrives. GAE therefore hands
			// the retreat a reliably negative advantage and the policy learns an
			// aversion to the PREFIX of the solution.
			//
			// Still potential-based: max-progress is part of the (augmented)
			// state and monotone, so the shaping telescopes to
			// K * (MaxPhi_final - MaxPhi_initial) exactly as before, and the
			// optimal policy is unchanged.
			//
			// Monotonicity is also what makes this safe. The tempting
			// alternative -- full credit forward, reduced penalty backward -- is
			// FARMABLE: oscillating over 32 units nets +0.021 per cycle against
			// ~0.018 of time penalty, so the agent can earn reward forever
			// without going anywhere. A ratchet can never re-earn ground.
			float Gain = Phi - State.m_MaxPhi;
			if(Gain > 0.0f)
			{
				Reward += SHAPING_K * Gain;
				State.m_MaxPhi = Phi;
			}
		}
		else
		{
			Reward += SHAPING_K * (SHAPING_GAMMA * Phi - State.m_LastPhi);
			// Tracked even when unused by the reward, so the observation field
			// and its logging mean the same thing in both arms of the A/B.
			State.m_MaxPhi = std::max(State.m_MaxPhi, Phi);
		}

		State.m_LastPhi = Phi;
	}

	if(m_FinishPos.x > 0 && distance(Pos, m_FinishPos) < 48.0f)
		Reward += FINISH_BONUS;

	State.m_LastPosX = Local.m_PosX;
	State.m_LastPosY = Local.m_PosY;
	State.m_LastHookState = Local.m_HookState;

	return Reward;
}

void CBotManager::ResetRewardState(int BotIndex, vec2 SpawnPos)
{
	SBotRewardState &State = m_aBotState[BotIndex];
	State.m_LastHookState = 0;
	State.m_WasDead = false;
	State.m_VisitedRegions.reset();
	State.m_NewRegionsThisEpisode = 0;
	State.m_LastPosX = SpawnPos.x;
	State.m_LastPosY = SpawnPos.y;
	State.m_HighestTimeCp = -1;

	// Seed the tracked index with an unrestricted scan — the one place a
	// backward jump is legitimate, because an episode boundary (or a restore
	// into an arbitrary mid-map state) breaks temporal continuity by design.
	if(m_WaypointsLoaded)
	{
		State.m_WaypointIndex = FindClosestWaypoint(SpawnPos, 0, (int)m_vWaypoints.size());
		State.m_LastPhi = GetPathProgress(SpawnPos, State.m_WaypointIndex) / m_TotalPathLength;
	}
	else
	{
		State.m_WaypointIndex = 0;
		State.m_LastPhi = 0.0f;
	}
	// The high-water mark starts wherever the episode starts, NOT at zero. A
	// curriculum restore drops the agent mid-map; seeding at 0 would hand it the
	// whole prefix as free ratchet reward for standing still.
	State.m_MaxPhi = State.m_LastPhi;
	// Seed the bin too, or the first tick after a restore snapshots a state we
	// just restored from and overwrites a good entry with a duplicate.
	State.m_LastBin = ProgressToBin(State.m_LastPhi);
}

void CBotManager::RespawnAllBots()
{
	for(int i = 0; i < m_NumBots; i++)
	{
		int ClientId = m_aClientIds[i];
		CPlayer *pPlayer = m_pGameContext->m_apPlayers[ClientId];
		if(pPlayer)
		{
			if(pPlayer->GetCharacter())
			{
				pPlayer->KillCharacter(WEAPON_GAME, false);
			}
			pPlayer->Respawn();
			ResetRewardState(i, GetSpawnPos());
		}
	}
}

vec2 CBotManager::GetSpawnPos()
{
	// Find spawn point from game controller
	vec2 SpawnPos(0, 0);
	if(m_pGameContext->m_pController)
	{
		m_pGameContext->m_pController->CanSpawn(0, &SpawnPos, -1);
	}
	return SpawnPos;
}

void CBotManager::AnalyzeMap()
{
	if(!m_pCollision)
		return;

	m_vCheckpoints.clear();
	m_SpawnPos = vec2(0, 0);
	m_FinishPos = vec2(0, 0);

	int MapWidth = m_pCollision->GetWidth();
	int MapHeight = m_pCollision->GetHeight();

	// Maps: tile position -> checkpoint index
	std::map<std::pair<int, int>, int> TileIndices;
	std::vector<std::pair<int, int>> CheckpointTiles;

	// Get front layer for time checkpoints
	const CTile *pFront = m_pCollision->FrontLayer();

	// Scan all tiles (both game layer and front layer)
	for(int y = 0; y < MapHeight; y++)
	{
		for(int x = 0; x < MapWidth; x++)
		{
			int MapIndex = y * MapWidth + x;
			int TileIdx = m_pCollision->GetTileIndex(MapIndex);

			// Check game layer
			if(TileIdx == TILE_START)
			{
				m_SpawnPos = vec2((x + 0.5f) * 32.0f, (y + 0.5f) * 32.0f);
			}
			else if(TileIdx == TILE_FINISH)
			{
				m_FinishPos = vec2((x + 0.5f) * 32.0f, (y + 0.5f) * 32.0f);
			}
			else if(TileIdx == TILE_CP || TileIdx == TILE_CP_F)
			{
				// Generic checkpoint (no index)
				CheckpointTiles.push_back(std::make_pair(x, y));
				TileIndices[std::make_pair(x, y)] = -1;
			}
			else if(TileIdx >= TILE_TIME_CHECKPOINT_FIRST && TileIdx <= TILE_TIME_CHECKPOINT_LAST)
			{
				// Time checkpoint with index (game layer)
				CheckpointTiles.push_back(std::make_pair(x, y));
				TileIndices[std::make_pair(x, y)] = TileIdx - TILE_TIME_CHECKPOINT_FIRST;
			}

			// Check front layer for time checkpoints
			if(pFront)
			{
				int FrontIdx = pFront[MapIndex].m_Index;
				if(FrontIdx >= TILE_TIME_CHECKPOINT_FIRST && FrontIdx <= TILE_TIME_CHECKPOINT_LAST)
				{
					auto Key = std::make_pair(x, y);
					if(TileIndices.find(Key) == TileIndices.end())
					{
						CheckpointTiles.push_back(Key);
						TileIndices[Key] = FrontIdx - TILE_TIME_CHECKPOINT_FIRST;
					}
				}
			}
		}
	}

	// Group adjacent tiles into checkpoints
	std::vector<bool> Visited(CheckpointTiles.size(), false);

	for(size_t i = 0; i < CheckpointTiles.size(); i++)
	{
		if(Visited[i])
			continue;

		// BFS to find connected tiles
		std::vector<std::pair<int, int>> Group;
		std::vector<size_t> Queue;
		Queue.push_back(i);
		Visited[i] = true;

		while(!Queue.empty())
		{
			size_t Idx = Queue.back();
			Queue.pop_back();
			Group.push_back(CheckpointTiles[Idx]);

			int cx = CheckpointTiles[Idx].first;
			int cy = CheckpointTiles[Idx].second;

			// Check adjacent tiles
			for(size_t j = 0; j < CheckpointTiles.size(); j++)
			{
				if(Visited[j])
					continue;

				int dx = std::abs(CheckpointTiles[j].first - cx);
				int dy = std::abs(CheckpointTiles[j].second - cy);

				if(dx <= 1 && dy <= 1)
				{
					Visited[j] = true;
					Queue.push_back(j);
				}
			}
		}

		// Calculate center and index of this checkpoint group
		float SumX = 0, SumY = 0;
		int MaxIndex = -1;

		for(const auto &Tile : Group)
		{
			SumX += (Tile.first + 0.5f) * 32.0f;
			SumY += (Tile.second + 0.5f) * 32.0f;

			auto It = TileIndices.find(Tile);
			if(It != TileIndices.end() && It->second > MaxIndex)
			{
				MaxIndex = It->second;
			}
		}

		SCheckpoint Cp;
		Cp.m_Pos = vec2(SumX / Group.size(), SumY / Group.size());
		Cp.m_Index = MaxIndex;
		m_vCheckpoints.push_back(Cp);
	}

	// Sort checkpoints by index
	std::sort(m_vCheckpoints.begin(), m_vCheckpoints.end(),
		[](const SCheckpoint &a, const SCheckpoint &b) {
			return a.m_Index < b.m_Index;
		});

	// Re-assign sequential indices
	for(size_t i = 0; i < m_vCheckpoints.size(); i++)
	{
		m_vCheckpoints[i].m_Index = (int)i;
	}

	m_MapAnalyzed = true;

	dbg_msg("rl/bots", "Map analyzed: %d time checkpoints, spawn (%.0f, %.0f), finish (%.0f, %.0f)",
		(int)m_vCheckpoints.size(), m_SpawnPos.x, m_SpawnPos.y, m_FinishPos.x, m_FinishPos.y);

	// Try to load waypoints for this map
	// Search paths: ./waypoints/<MapName>.waypoints.bin, ./ghosts/<MapName>.waypoints.bin
	const char *pMapName = m_pGameContext->Server()->GetMapName();
	char aPath[256];
	str_format(aPath, sizeof(aPath), "waypoints/%s.waypoints.bin", pMapName);
	if(!LoadWaypoints(aPath))
	{
		str_format(aPath, sizeof(aPath), "ghosts/%s.waypoints.bin", pMapName);
		LoadWaypoints(aPath);
	}
}

vec2 CBotManager::GetNextCheckpointPos(int CurrentIndex) const
{
	int NextIdx = CurrentIndex + 1;

	// Try teleport checkpoint first (more common in DDNet maps)
	// TeleCheckOuts uses 0-indexed checkpoint numbers
	const auto &TeleOuts = m_pCollision->TeleCheckOuts(NextIdx);
	if(!TeleOuts.empty())
	{
		return TeleOuts[0];  // Return first teleport-out position
	}

	// Fall back to time checkpoint array
	if(NextIdx >= 0 && NextIdx < (int)m_vCheckpoints.size())
	{
		return m_vCheckpoints[NextIdx].m_Pos;
	}

	// No more checkpoints, target finish
	return m_FinishPos.x > 0 ? m_FinishPos : m_SpawnPos;
}

int CBotManager::GetCheckpointAtPos(vec2 Pos, float Radius) const
{
	for(const auto &Cp : m_vCheckpoints)
	{
		if(distance(Pos, Cp.m_Pos) < Radius)
		{
			return Cp.m_Index;
		}
	}
	return -1;
}

bool CBotManager::LoadWaypoints(const std::string &Path)
{
	FILE *pFile = fopen(Path.c_str(), "rb");
	if(!pFile)
	{
		dbg_msg("rl/bots", "No waypoints file found at '%s'", Path.c_str());
		return false;
	}

	uint32_t NumPoints;
	if(fread(&NumPoints, sizeof(uint32_t), 1, pFile) != 1 || NumPoints == 0 || NumPoints > 100000)
	{
		dbg_msg("rl/bots", "Invalid waypoints file: bad point count %u", NumPoints);
		fclose(pFile);
		return false;
	}

	m_vWaypoints.resize(NumPoints);
	float CumulDist = 0.0f;
	float LastX = 0, LastY = 0;

	for(uint32_t i = 0; i < NumPoints; i++)
	{
		float x, y;
		if(fread(&x, sizeof(float), 1, pFile) != 1 || fread(&y, sizeof(float), 1, pFile) != 1)
		{
			dbg_msg("rl/bots", "Waypoints file truncated at point %u", i);
			fclose(pFile);
			m_vWaypoints.clear();
			return false;
		}

		if(i > 0)
		{
			float dx = x - LastX;
			float dy = y - LastY;
			CumulDist += sqrtf(dx * dx + dy * dy);
		}

		m_vWaypoints[i].m_X = x;
		m_vWaypoints[i].m_Y = y;
		m_vWaypoints[i].m_CumulativeDist = CumulDist;
		LastX = x;
		LastY = y;
	}

	fclose(pFile);
	m_TotalPathLength = CumulDist;
	m_WaypointsLoaded = true;

	dbg_msg("rl/bots", "Loaded %u waypoints, total path length: %.0f units",
		NumPoints, m_TotalPathLength);

	SegmentPathAtDiscontinuities();

	// Archive is progress-binned, so it only makes sense once a path exists.
	m_vSnapshots.assign(NUM_PROGRESS_BINS, CSaveTee());
	m_vSnapshotValid.assign(NUM_PROGRESS_BINS, 0);
	m_SnapshotsCaptured = 0;

	UpdateMatchWindow();
	return true;
}

// Size the forward search window from physics, not from a fixed constant. The
// tee cannot cover more than MAX_TEE_SPEED * FrameSkip units of arc between two
// calls, so anything beyond that is a wrong-strand match. Called from both
// LoadWaypoints and SetFrameSkip, since either can arrive first.
void CBotManager::UpdateMatchWindow()
{
	if(!m_WaypointsLoaded || m_vWaypoints.size() < 2)
		return;

	float Spacing = m_TotalPathLength / (float)(m_vWaypoints.size() - 1);
	m_MaxAdvance = (int)ceilf(MAX_TEE_SPEED * (float)std::max(1, m_FrameSkip) / std::max(Spacing, 1.0f)) + 1;
	dbg_msg("rl/bots", "Waypoint spacing %.0f units, match window +%d/-%d (frame skip %d)",
		Spacing, m_MaxAdvance, MAX_RETREAT, m_FrameSkip);
}

// Temporally coherent nearest-waypoint search.
//
// A purely spatial window is not enough. Reference paths double back on
// themselves at close range — on Tutorial the ghost passes within ~100 units of
// a point ~1000 units later along its own path, only ~17 waypoints apart, well
// inside the +/-100 window this used to use. Measured against ground truth
// (waypoints are resampled from the ghost, so cumulative ghost length IS the
// true arc position), the old window gave a max error of 1.8% of the map and
// single-frame jumps of 1052 units; this gives 0.45% and 355 units, the latter
// being the resampling floor. With a high-water-mark reward that error was
// permanent; with potential shaping it is merely noise, but it is still wrong.
//
// SearchRadius is the forward allowance. Pass m_vWaypoints.size() to opt out
// entirely — legitimate only at an episode boundary, where temporal continuity
// is broken by design.
int CBotManager::FindClosestWaypoint(vec2 Pos, int SearchStart, int SearchRadius) const
{
	if(m_vWaypoints.empty())
		return -1;

	const int NumWaypoints = (int)m_vWaypoints.size();
	const bool Unrestricted = SearchRadius >= NumWaypoints;
	SearchStart = std::clamp(SearchStart, 0, NumWaypoints - 1);

	int Start = Unrestricted ? 0 : std::max(0, SearchStart - MAX_RETREAT);
	int End = Unrestricted ? NumWaypoints : std::min(NumWaypoints, SearchStart + SearchRadius + 1);

	int BestIdx = SearchStart;
	float BestDist = 1e18f;
	for(int i = Start; i < End; i++)
	{
		float dx = Pos.x - m_vWaypoints[i].m_X;
		float dy = Pos.y - m_vWaypoints[i].m_Y;
		float d = dx * dx + dy * dy;
		if(d < BestDist)
		{
			BestDist = d;
			BestIdx = i;
		}
	}

	// Lost: teleported, or wandered far off the path. Re-acquire by scanning
	// FORWARD only. A backward global snap mid-episode is always wrong — it lets
	// the match land on an earlier strand of a path that doubles back.
	if(!Unrestricted && BestDist > LOST_DIST * LOST_DIST)
	{
		for(int i = SearchStart; i < NumWaypoints; i++)
		{
			float dx = Pos.x - m_vWaypoints[i].m_X;
			float dy = Pos.y - m_vWaypoints[i].m_Y;
			float d = dx * dx + dy * dy;
			if(d < BestDist)
			{
				BestDist = d;
				BestIdx = i;
			}
		}
	}

	return BestIdx;
}

// Split the path wherever consecutive waypoints are not actually connected —
// i.e. a teleport. resample_waypoints() walks the ghost polyline at fixed
// spacing and happily interpolates straight across a teleport, laying phantom
// waypoints through solid geometry and giving two physically distant regions
// adjacent indices.
//
// Detection is by collision, not by distance: free fall legitimately reaches
// ~72 units/tick and ghosts record at 25 Hz, so large gaps between consecutive
// samples are normal, but a segment passing through solid tiles is not.
void CBotManager::SegmentPathAtDiscontinuities()
{
	int SegmentId = 0;
	int NumBreaks = 0;
	if(!m_vWaypoints.empty())
		m_vWaypoints[0].m_SegmentId = 0;

	for(size_t i = 1; i < m_vWaypoints.size(); i++)
	{
		bool Break = false;
		if(m_pCollision)
		{
			vec2 A(m_vWaypoints[i - 1].m_X, m_vWaypoints[i - 1].m_Y);
			vec2 B(m_vWaypoints[i].m_X, m_vWaypoints[i].m_Y);
			// Sample along the segment; any solid tile means these two points are
			// not reachable from one another in a straight line.
			float Len = distance(A, B);
			int Steps = std::max(2, (int)(Len / 16.0f));
			for(int s = 1; s < Steps; s++)
			{
				vec2 P = mix(A, B, (float)s / (float)Steps);
				if(m_pCollision->CheckPoint(P.x, P.y))
				{
					Break = true;
					break;
				}
			}
		}
		if(Break)
		{
			SegmentId++;
			NumBreaks++;
			// The portal edge contributes ZERO arc length — no free progress
			// reward for the distance a teleport covers.
			float PrevCumul = m_vWaypoints[i - 1].m_CumulativeDist;
			float Delta = m_vWaypoints[i].m_CumulativeDist - PrevCumul;
			for(size_t j = i; j < m_vWaypoints.size(); j++)
				m_vWaypoints[j].m_CumulativeDist -= Delta;
		}
		m_vWaypoints[i].m_SegmentId = SegmentId;
	}

	if(NumBreaks > 0)
	{
		m_TotalPathLength = m_vWaypoints.back().m_CumulativeDist;
		dbg_msg("rl/bots", "Path split into %d segments at %d discontinuity/ies, "
			"corrected path length: %.0f units", SegmentId + 1, NumBreaks, m_TotalPathLength);
	}
}

float CBotManager::GetPathProgress(vec2 Pos, int WaypointIdx) const
{
	if(m_vWaypoints.empty() || WaypointIdx < 0)
		return 0.0f;

	WaypointIdx = std::clamp(WaypointIdx, 0, (int)m_vWaypoints.size() - 1);

	// Base progress = cumulative distance to this waypoint
	float Progress = m_vWaypoints[WaypointIdx].m_CumulativeDist;

	// Add fractional progress toward next waypoint
	if(WaypointIdx + 1 < (int)m_vWaypoints.size())
	{
		vec2 WpCur(m_vWaypoints[WaypointIdx].m_X, m_vWaypoints[WaypointIdx].m_Y);
		vec2 WpNext(m_vWaypoints[WaypointIdx + 1].m_X, m_vWaypoints[WaypointIdx + 1].m_Y);
		vec2 Seg = WpNext - WpCur;
		float SegLen = length(Seg);
		if(SegLen > 0.1f)
		{
			vec2 ToPos = Pos - WpCur;
			float Proj = dot(ToPos, Seg) / (SegLen * SegLen);
			Proj = std::clamp(Proj, 0.0f, 1.0f);
			Progress += Proj * SegLen;
		}
	}

	return Progress;
}

void CBotManager::GetStats(int *pTotalTicks, int *pMissedActions, int *pMaxConsecutive, int64_t *pMaxLatencyUs) const
{
	if(pTotalTicks)
		*pTotalTicks = m_TotalTicks;
	if(pMissedActions)
		*pMissedActions = m_MissedActions;
	if(pMaxConsecutive)
		*pMaxConsecutive = m_MaxConsecutiveMisses;
	if(pMaxLatencyUs)
		*pMaxLatencyUs = m_MaxLatencyUs;
}

void CBotManager::SetFrameSkip(int FrameSkip)
{
	if(FrameSkip < 1)
		FrameSkip = 1;
	if(FrameSkip > 100)
		FrameSkip = 100;

	m_FrameSkip = FrameSkip;
	m_FrameSkipCounter = 0; // Reset counter
	UpdateMatchWindow();  // window depends on frame skip; either may be set first
	mem_zero(m_aAccumulatedRewards, sizeof(m_aAccumulatedRewards)); // Clear accumulated rewards

	dbg_msg("rl/bots", "Frame skip set to %d (decision rate: server_tickrate/%d)", FrameSkip, FrameSkip);
}

bool CBotManager::IsPaused() const
{
	if(!m_Enabled || !m_pShm || !m_AllowPause)
		return false;

	return m_pShm->IsPauseRequested();
}

} // namespace rl
