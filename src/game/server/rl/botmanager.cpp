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
	m_LastTick(0)
{
	mem_zero(m_aClientIds, sizeof(m_aClientIds));
	mem_zero(m_aBotState, sizeof(m_aBotState));
	mem_zero(m_aBotInputs, sizeof(m_aBotInputs));
}

CBotManager::~CBotManager()
{
	Shutdown();
}

bool CBotManager::Init(int NumBots)
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
	if(!m_pShm->Init(NumBots))
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
			str_format(aBotName, sizeof(aBotName), "RL Bot %d", i);
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
		ResetRewardState(i);

		// Store bot client ID in shared memory
		SBotData *pBotData = m_pShm->GetBotData(i);
		if(pBotData)
		{
			pBotData->m_BotClientId = NextClientId;
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
				str_format(aBotName, sizeof(aBotName), "RL Bot %d", i);
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

	// Check if Python has actions ready (non-blocking)
	// We use a very short timeout to avoid blocking the server's network processing
	// This allows real clients to connect and watch the bots
	if(!m_pShm->AreActionsReady())
	{
		// No actions ready - use previous inputs
		// Don't block - the server needs to process network events
		return;
	}

	// Inject inputs
	InjectBotInputs();

	// Clear action ready flag
	m_pShm->ClearActionsReady();

	// Extract observations after physics update (will be called after world tick)
}

void CBotManager::InjectBotInputs()
{
	for(int i = 0; i < m_NumBots; i++)
	{
		int ClientId = m_aClientIds[i];

		// Get action from shared memory
		SBotData *pBotData = m_pShm->GetBotData(i);
		if(!pBotData)
			continue;

		// Check if we have a real player for this bot
		CPlayer *pPlayer = m_pGameContext->m_apPlayers[ClientId];

		// Handle reset request
		if(pBotData->m_Action.m_Reset)
		{
			if(pPlayer && pPlayer->GetCharacter())
			{
				pPlayer->KillCharacter(WEAPON_GAME, false);
			}
			if(pPlayer)
			{
				// Use ForceSpawn instead of Respawn for bots
				// Respawn() just sets m_Spawning=true, but TryRespawn() checks
				// Server()->ClientIngame() which returns false for bots
				vec2 SpawnPos = GetSpawnPos();
				pPlayer->ForceSpawn(SpawnPos);
			}
			ResetRewardState(i);
			pBotData->m_Action.m_Reset = 0;
		}

		// Convert action to input
		ConvertActionToInput(&pBotData->m_Action, &m_aBotInputs[i]);

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

	// Analyze map for checkpoints on first call
	if(!m_MapAnalyzed && m_pCollision)
	{
		AnalyzeMap();
	}

	int CurrentTick = m_pGameContext->Server()->Tick();

	for(int i = 0; i < m_NumBots; i++)
	{
		int ClientId = m_aClientIds[i];
		SBotData *pBotData = m_pShm->GetBotData(i);

		if(!pBotData)
			continue;

		SStructuredObservation *pObs = &pBotData->m_Observation;
		mem_zero(pObs, sizeof(SStructuredObservation));

		pObs->m_Tick = CurrentTick;

		// Check if we have a real player with character
		CPlayer *pPlayer = m_pGameContext->m_apPlayers[ClientId];
		CCharacter *pChar = pPlayer ? pPlayer->GetCharacter() : nullptr;

		if(pChar && pChar->IsAlive())
		{
			// Extract local player state
			ExtractPlayerObs(pChar, &pObs->m_LocalPlayer, true);

			// Extract tile grid
			ExtractTileGrid(pChar, pObs->m_aTileGrid);

			// Extract raycasts
			ExtractRaycasts(pChar, pObs);

			// Extract other players
			int OtherIdx = 0;
			for(int j = 0; j < MAX_CLIENTS && OtherIdx < MAX_OTHER_PLAYERS; j++)
			{
				if(j == ClientId)
					continue;

				CPlayer *pOther = m_pGameContext->m_apPlayers[j];
				if(pOther && pOther->GetCharacter())
				{
					ExtractPlayerObs(pOther->GetCharacter(), &pObs->m_aOtherPlayers[OtherIdx], false);
					OtherIdx++;
				}
			}
			pObs->m_NumActivePlayers = OtherIdx + 1;

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

			// Checkpoint navigation info
			int CurrentCpIdx = m_aBotState[i].m_CurrentCheckpointIndex;
			vec2 NextCpPos = GetNextCheckpointPos(CurrentCpIdx);
			pObs->m_NextCheckpointX = NextCpPos.x;
			pObs->m_NextCheckpointY = NextCpPos.y;
			pObs->m_CheckpointIndex = CurrentCpIdx;
			pObs->m_TotalCheckpoints = (int)m_vCheckpoints.size();

			// Check if bot reached a new checkpoint
			vec2 CharPos = pChar->GetPos();
			int CpAtPos = GetCheckpointAtPos(CharPos);
			if(CpAtPos >= 0 && CpAtPos > CurrentCpIdx)
			{
				m_aBotState[i].m_CurrentCheckpointIndex = CpAtPos;
			}

			// Calculate reward
			pBotData->m_Reward = CalculateReward(i, pObs);
			pBotData->m_Done = 0;
		}
		else
		{
			// Character dead or not spawned
			pBotData->m_Reward = -1.0f; // Death penalty
			pBotData->m_Done = 1;
		}
	}

	// Update header tick
	SServerRLHeader *pHeader = m_pShm->GetHeader();
	if(pHeader)
	{
		pHeader->m_ServerTick = CurrentTick;
	}

	// Signal observations are ready
	m_pShm->SignalObservationsReady();

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

	for(int gy = 0; gy < TILE_GRID_HEIGHT; gy++)
	{
		for(int gx = 0; gx < TILE_GRID_WIDTH; gx++)
		{
			int TileX = PlayerTileX + (gx - HalfWidth);
			int TileY = PlayerTileY + (gy - HalfHeight);

			uint8_t TileType = (uint8_t)ETileType::EMPTY;

			if(TileX >= 0 && TileX < MapWidth && TileY >= 0 && TileY < MapHeight)
			{
				int MapIndex = TileY * MapWidth + TileX;
				int TileIdx = m_pCollision->GetTileIndex(MapIndex);

				switch(TileIdx)
				{
				case TILE_AIR: TileType = (uint8_t)ETileType::EMPTY; break;
				case TILE_SOLID: TileType = (uint8_t)ETileType::SOLID; break;
				case TILE_DEATH: TileType = (uint8_t)ETileType::DEATH; break;
				case TILE_NOHOOK: TileType = (uint8_t)ETileType::NOHOOK; break;
				case TILE_FREEZE: TileType = (uint8_t)ETileType::FREEZE; break;
				case TILE_UNFREEZE: TileType = (uint8_t)ETileType::UNFREEZE; break;
				case TILE_START: TileType = (uint8_t)ETileType::SPAWN; break;
				case TILE_FINISH: TileType = (uint8_t)ETileType::FINISH; break;
				case TILE_CP:
				case TILE_CP_F: TileType = (uint8_t)ETileType::CHECKPOINT; break;
				default:
					if(TileIdx >= TILE_TIME_CHECKPOINT_FIRST && TileIdx <= TILE_TIME_CHECKPOINT_LAST)
						TileType = (uint8_t)ETileType::CHECKPOINT;
					else if(TileIdx > 0)
						TileType = (uint8_t)ETileType::OTHER;
					break;
				}
			}
			else
			{
				TileType = (uint8_t)ETileType::SOLID; // Out of bounds
			}

			pGrid[gy * TILE_GRID_WIDTH + gx] = TileType;
		}
	}
}

void CBotManager::ExtractRaycasts(CCharacter *pChar, SStructuredObservation *pObs)
{
	if(!pChar)
		return;

	vec2 Pos = pChar->GetPos();

	for(int i = 0; i < NUM_RAYCASTS; i++)
	{
		float Angle = (float)i * (2.0f * 3.14159265f / NUM_RAYCASTS);
		vec2 Dir(std::cos(Angle), std::sin(Angle));

		pObs->m_aRaycastSolid[i] = CastRay(Pos, Dir, RAYCAST_MAX_DIST, 1);
		pObs->m_aRaycastDeath[i] = CastRay(Pos, Dir, RAYCAST_MAX_DIST, 2);
		pObs->m_aRaycastHookable[i] = CastRay(Pos, Dir, RAYCAST_MAX_DIST, 4);
	}
}

float CBotManager::CastRay(vec2 Start, vec2 Dir, float MaxDist, int TileMask)
{
	if(!m_pCollision)
		return 1.0f;

	int MapWidth = m_pCollision->GetWidth();
	int MapHeight = m_pCollision->GetHeight();

	const float Step = 16.0f;
	for(float Dist = Step; Dist < MaxDist; Dist += Step)
	{
		vec2 Pos = Start + Dir * Dist;
		int TileX = (int)(Pos.x / 32.0f);
		int TileY = (int)(Pos.y / 32.0f);

		if(TileX < 0 || TileX >= MapWidth || TileY < 0 || TileY >= MapHeight)
			return Dist / MaxDist;

		int MapIndex = TileY * MapWidth + TileX;
		int TileIdx = m_pCollision->GetTileIndex(MapIndex);

		if((TileMask & 1) && TileIdx == TILE_SOLID)
			return Dist / MaxDist;
		if((TileMask & 2) && TileIdx == TILE_DEATH)
			return Dist / MaxDist;
		if((TileMask & 4) && (TileIdx == TILE_SOLID || TileIdx == TILE_NOHOOK))
			return Dist / MaxDist;
	}

	return 1.0f;
}

float CBotManager::CalculateReward(int BotIndex, const SStructuredObservation *pObs)
{
	float Reward = 0.0f;
	SBotRewardState &State = m_aBotState[BotIndex];
	const SPlayerObs &Local = pObs->m_LocalPlayer;

	if(!Local.m_Active)
		return 0.0f;

	vec2 Pos(Local.m_PosX, Local.m_PosY);
	vec2 NextCp(pObs->m_NextCheckpointX, pObs->m_NextCheckpointY);

	// Calculate distance to next checkpoint
	vec2 ToCheckpoint = NextCp - Pos;
	float DistToCheckpoint = length(ToCheckpoint);

	// Progress toward next checkpoint (direction-agnostic dense reward)
	float Progress = State.m_LastDistToCheckpoint - DistToCheckpoint;
	Reward += Progress / 100.0f;

	// Checkpoint reached (sparse but important)
	if(State.m_CurrentCheckpointIndex > pObs->m_CheckpointIndex)
	{
		// This was already updated in ExtractObservations, so check against stored state
	}
	int CurrentCp = m_aBotState[BotIndex].m_CurrentCheckpointIndex;
	static int s_aLastCp[MAX_RL_BOTS] = {-1};
	if(CurrentCp > s_aLastCp[BotIndex])
	{
		Reward += 10.0f;
		s_aLastCp[BotIndex] = CurrentCp;
		// Reset distance tracking for new checkpoint
		vec2 NewNextCp = GetNextCheckpointPos(CurrentCp);
		State.m_LastDistToCheckpoint = length(NewNextCp - Pos);
	}

	// Finish reached
	if(m_FinishPos.x > 0 && distance(Pos, m_FinishPos) < 48.0f)
	{
		Reward += 100.0f;
	}

	// Time penalty (small)
	Reward -= 0.0005f;

	// Velocity bonus: reward movement TOWARD checkpoint
	if(DistToCheckpoint > 0.1f)
	{
		vec2 DirToCheckpoint = ToCheckpoint / DistToCheckpoint;
		vec2 Vel(Local.m_VelX, Local.m_VelY);
		float VelTowardCp = dot(Vel, DirToCheckpoint);
		Reward += std::clamp(VelTowardCp, 0.0f, 20.0f) / 500.0f;
	}

	// Hook success
	if(Local.m_HookState == 2 && State.m_LastHookState != 2)
	{
		Reward += 0.05f;
	}

	// Freeze penalty
	if(Local.m_IsFrozen && !State.m_WasFrozen)
	{
		Reward -= 0.3f;
	}
	if(Local.m_IsFrozen)
	{
		Reward -= 0.1f; // Per-tick penalty while frozen
	}

	// Stuck penalty
	if(std::abs(Local.m_PosX - State.m_LastPosX) < 1.0f &&
	   std::abs(Local.m_PosY - State.m_LastPosY) < 1.0f)
	{
		State.m_StuckCounter++;
		if(State.m_StuckCounter > 100)
		{
			Reward -= 0.01f;
		}
	}
	else
	{
		State.m_StuckCounter = 0;
	}

	// Update state
	State.m_LastPosX = Local.m_PosX;
	State.m_LastPosY = Local.m_PosY;
	State.m_LastDistToCheckpoint = DistToCheckpoint;
	State.m_LastHookState = Local.m_HookState;
	State.m_WasFrozen = Local.m_IsFrozen;

	return Reward;
}

void CBotManager::ResetRewardState(int BotIndex)
{
	SBotRewardState &State = m_aBotState[BotIndex];
	State.m_LastPosX = m_SpawnPos.x;
	State.m_LastPosY = m_SpawnPos.y;
	State.m_CurrentCheckpointIndex = -1;
	State.m_LastDistToCheckpoint = 10000.0f; // Will be set properly on first tick
	State.m_LastHookState = 0;
	State.m_StuckCounter = 0;
	State.m_WasFrozen = false;
	State.m_WasDead = false;
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
			ResetRewardState(i);
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

	dbg_msg("rl/bots", "Map analyzed: %d checkpoints, spawn (%.0f, %.0f), finish (%.0f, %.0f)",
		(int)m_vCheckpoints.size(), m_SpawnPos.x, m_SpawnPos.y, m_FinishPos.x, m_FinishPos.y);
}

vec2 CBotManager::GetNextCheckpointPos(int CurrentIndex) const
{
	// If no checkpoints, target finish
	if(m_vCheckpoints.empty())
	{
		return m_FinishPos.x > 0 ? m_FinishPos : m_SpawnPos;
	}

	// If before any checkpoint, target first
	if(CurrentIndex < 0)
	{
		return m_vCheckpoints[0].m_Pos;
	}

	// If at or past last checkpoint, target finish
	if(CurrentIndex >= (int)m_vCheckpoints.size() - 1)
	{
		return m_FinishPos.x > 0 ? m_FinishPos : m_vCheckpoints.back().m_Pos;
	}

	// Target next checkpoint
	return m_vCheckpoints[CurrentIndex + 1].m_Pos;
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

} // namespace rl
