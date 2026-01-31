/* Teeworlds RL Bot - Map Analysis for Checkpoint Extraction */
#ifndef GAME_CLIENT_RL_MAP_ANALYSIS_H
#define GAME_CLIENT_RL_MAP_ANALYSIS_H

#include <base/vmath.h>
#include <vector>
#include <map>

// Forward declarations
class CCollision;
class CLayers;

namespace rl {

// Represents a checkpoint or waypoint in the map
struct SCheckpoint
{
	vec2 m_Pos;           // World position (center of checkpoint tiles)
	int m_Index;          // Checkpoint index (0-based)
	int m_TileIndex;      // Original tile index (for time checkpoints: 35-59)
	bool m_IsTimeCheckpoint; // True if this is a time checkpoint (has index in tile)

	bool operator<(const SCheckpoint &Other) const
	{
		return m_Index < Other.m_Index;
	}
};

// Analyzes DDNet maps to extract checkpoint positions and order
class CMapAnalysis
{
public:
	CMapAnalysis();
	~CMapAnalysis();

	// Analyze a map for checkpoints, spawn, and finish positions
	// Must be called after map is loaded
	void AnalyzeMap(CCollision *pCollision, CLayers *pLayers);

	// Clear all analysis data
	void Clear();

	// Get number of checkpoints found
	int GetNumCheckpoints() const { return (int)m_vCheckpoints.size(); }

	// Get checkpoint by index (0-based)
	const SCheckpoint *GetCheckpoint(int Index) const;

	// Get position of next checkpoint after given index
	// Returns finish position if at last checkpoint
	// Returns spawn if index is -1 (not started)
	vec2 GetNextCheckpointPos(int CurrentIndex) const;

	// Get the checkpoint index at a given position (or -1 if none)
	int GetCheckpointAtPos(vec2 Pos, float Radius = 32.0f) const;

	// Get spawn position (start tile)
	vec2 GetSpawnPos() const { return m_SpawnPos; }

	// Get finish position
	vec2 GetFinishPos() const { return m_FinishPos; }

	// Check if map has been analyzed
	bool IsAnalyzed() const { return m_Analyzed; }

	// Get all checkpoints (for iteration)
	const std::vector<SCheckpoint> &GetCheckpoints() const { return m_vCheckpoints; }

private:
	// Group adjacent checkpoint tiles into single checkpoints
	void GroupCheckpointTiles(const std::vector<std::pair<int, int>> &TilePositions,
	                         const std::map<std::pair<int, int>, int> &TileIndices);

	// Flood fill to find connected tiles
	void FloodFillCheckpoint(int StartX, int StartY,
	                        const std::vector<std::pair<int, int>> &AllTiles,
	                        std::vector<bool> &Visited,
	                        std::vector<std::pair<int, int>> &Connected);

	std::vector<SCheckpoint> m_vCheckpoints;
	vec2 m_SpawnPos;
	vec2 m_FinishPos;
	bool m_Analyzed;
};

} // namespace rl

#endif // GAME_CLIENT_RL_MAP_ANALYSIS_H
