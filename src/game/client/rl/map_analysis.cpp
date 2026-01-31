/* Teeworlds RL Bot - Map Analysis Implementation */
#include "map_analysis.h"

#include <base/system.h>
#include <game/collision.h>
#include <game/layers.h>
#include <game/mapitems.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <queue>

namespace rl {

CMapAnalysis::CMapAnalysis() :
	m_SpawnPos(0.0f, 0.0f),
	m_FinishPos(0.0f, 0.0f),
	m_Analyzed(false)
{
}

CMapAnalysis::~CMapAnalysis()
{
	Clear();
}

void CMapAnalysis::Clear()
{
	m_vCheckpoints.clear();
	m_SpawnPos = vec2(0.0f, 0.0f);
	m_FinishPos = vec2(0.0f, 0.0f);
	m_Analyzed = false;
}

void CMapAnalysis::AnalyzeMap(CCollision *pCollision, CLayers *pLayers)
{
	Clear();

	if(!pCollision || !pLayers)
		return;

	int MapWidth = pCollision->GetWidth();
	int MapHeight = pCollision->GetHeight();

	if(MapWidth <= 0 || MapHeight <= 0)
		return;

	// Collect all checkpoint tiles, spawn, and finish
	std::vector<std::pair<int, int>> CheckpointTiles;
	std::map<std::pair<int, int>, int> TileIndices; // Map position to tile index

	vec2 SpawnSum(0.0f, 0.0f);
	int SpawnCount = 0;
	vec2 FinishSum(0.0f, 0.0f);
	int FinishCount = 0;

	for(int y = 0; y < MapHeight; y++)
	{
		for(int x = 0; x < MapWidth; x++)
		{
			int MapIndex = y * MapWidth + x;
			int TileIdx = pCollision->GetTileIndex(MapIndex);

			// World position (center of tile)
			float WorldX = (x + 0.5f) * 32.0f;
			float WorldY = (y + 0.5f) * 32.0f;

			// Check for start/spawn tile
			if(TileIdx == TILE_START)
			{
				SpawnSum.x += WorldX;
				SpawnSum.y += WorldY;
				SpawnCount++;
			}
			// Check for finish tile
			else if(TileIdx == TILE_FINISH)
			{
				FinishSum.x += WorldX;
				FinishSum.y += WorldY;
				FinishCount++;
			}
			// Check for checkpoint tiles
			else if(TileIdx == TILE_CP || TileIdx == TILE_CP_F)
			{
				CheckpointTiles.push_back(std::make_pair(x, y));
				TileIndices[std::make_pair(x, y)] = 0; // Generic checkpoint, index 0
			}
			// Check for time checkpoints (35-59 have embedded index)
			else if(TileIdx >= TILE_TIME_CHECKPOINT_FIRST && TileIdx <= TILE_TIME_CHECKPOINT_LAST)
			{
				CheckpointTiles.push_back(std::make_pair(x, y));
				// Time checkpoint index is embedded in tile (35 = cp 0, 36 = cp 1, etc.)
				TileIndices[std::make_pair(x, y)] = TileIdx - TILE_TIME_CHECKPOINT_FIRST;
			}
		}
	}

	// Calculate average spawn position
	if(SpawnCount > 0)
	{
		m_SpawnPos = SpawnSum / (float)SpawnCount;
	}

	// Calculate average finish position
	if(FinishCount > 0)
	{
		m_FinishPos = FinishSum / (float)FinishCount;
	}

	// Group adjacent checkpoint tiles into single checkpoints
	if(!CheckpointTiles.empty())
	{
		GroupCheckpointTiles(CheckpointTiles, TileIndices);
	}

	// Sort checkpoints by index
	std::sort(m_vCheckpoints.begin(), m_vCheckpoints.end());

	m_Analyzed = true;

	dbg_msg("rl/map", "Map analysis complete: %d checkpoints, spawn=(%.0f,%.0f), finish=(%.0f,%.0f)",
	        (int)m_vCheckpoints.size(),
	        m_SpawnPos.x, m_SpawnPos.y,
	        m_FinishPos.x, m_FinishPos.y);
}

void CMapAnalysis::GroupCheckpointTiles(const std::vector<std::pair<int, int>> &TilePositions,
                                        const std::map<std::pair<int, int>, int> &TileIndices)
{
	// Create a set for fast lookup
	std::set<std::pair<int, int>> TileSet(TilePositions.begin(), TilePositions.end());
	std::vector<bool> Visited(TilePositions.size(), false);

	// Create position to index mapping
	std::map<std::pair<int, int>, int> PosToIdx;
	for(size_t i = 0; i < TilePositions.size(); i++)
	{
		PosToIdx[TilePositions[i]] = (int)i;
	}

	int NextAutoIndex = 0; // For checkpoints without embedded index

	for(size_t i = 0; i < TilePositions.size(); i++)
	{
		if(Visited[i])
			continue;

		// BFS to find all connected checkpoint tiles
		std::vector<std::pair<int, int>> Connected;
		std::queue<std::pair<int, int>> Queue;
		Queue.push(TilePositions[i]);
		Visited[i] = true;

		while(!Queue.empty())
		{
			auto Current = Queue.front();
			Queue.pop();
			Connected.push_back(Current);

			// Check 4 neighbors
			const int dx[] = {-1, 1, 0, 0};
			const int dy[] = {0, 0, -1, 1};
			for(int d = 0; d < 4; d++)
			{
				std::pair<int, int> Neighbor = std::make_pair(Current.first + dx[d], Current.second + dy[d]);
				if(TileSet.find(Neighbor) != TileSet.end())
				{
					auto it = PosToIdx.find(Neighbor);
					if(it != PosToIdx.end() && !Visited[it->second])
					{
						Visited[it->second] = true;
						Queue.push(Neighbor);
					}
				}
			}
		}

		// Calculate center position of this checkpoint group
		vec2 Center(0.0f, 0.0f);
		int MaxTileIndex = -1;
		bool HasTimeCheckpoint = false;

		for(const auto &Pos : Connected)
		{
			Center.x += (Pos.first + 0.5f) * 32.0f;
			Center.y += (Pos.second + 0.5f) * 32.0f;

			auto it = TileIndices.find(Pos);
			if(it != TileIndices.end())
			{
				if(it->second > 0)
				{
					// Time checkpoint with embedded index
					HasTimeCheckpoint = true;
					MaxTileIndex = std::max(MaxTileIndex, it->second);
				}
			}
		}
		Center /= (float)Connected.size();

		// Create checkpoint
		SCheckpoint Cp;
		Cp.m_Pos = Center;
		Cp.m_IsTimeCheckpoint = HasTimeCheckpoint;

		if(HasTimeCheckpoint)
		{
			Cp.m_Index = MaxTileIndex;
			Cp.m_TileIndex = MaxTileIndex + TILE_TIME_CHECKPOINT_FIRST;
		}
		else
		{
			// Generic checkpoint - assign sequential index
			Cp.m_Index = NextAutoIndex++;
			Cp.m_TileIndex = TILE_CP;
		}

		m_vCheckpoints.push_back(Cp);
	}
}

const SCheckpoint *CMapAnalysis::GetCheckpoint(int Index) const
{
	for(const auto &Cp : m_vCheckpoints)
	{
		if(Cp.m_Index == Index)
			return &Cp;
	}
	return nullptr;
}

vec2 CMapAnalysis::GetNextCheckpointPos(int CurrentIndex) const
{
	// If haven't started (index -1), return first checkpoint or finish if no checkpoints
	if(CurrentIndex < 0)
	{
		if(!m_vCheckpoints.empty())
			return m_vCheckpoints[0].m_Pos;
		return m_FinishPos;
	}

	// Find next checkpoint after current index
	for(const auto &Cp : m_vCheckpoints)
	{
		if(Cp.m_Index > CurrentIndex)
			return Cp.m_Pos;
	}

	// No more checkpoints, return finish
	return m_FinishPos;
}

int CMapAnalysis::GetCheckpointAtPos(vec2 Pos, float Radius) const
{
	float RadiusSq = Radius * Radius;
	for(const auto &Cp : m_vCheckpoints)
	{
		float DistSq = (Cp.m_Pos.x - Pos.x) * (Cp.m_Pos.x - Pos.x) +
		               (Cp.m_Pos.y - Pos.y) * (Cp.m_Pos.y - Pos.y);
		if(DistSq <= RadiusSq)
			return Cp.m_Index;
	}
	return -1;
}

} // namespace rl
