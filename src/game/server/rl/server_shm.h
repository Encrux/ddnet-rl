/* DDNet Server RL - Shared Memory Communication
 *
 * Handles batched communication between Python and the DDNet server
 * for multiple bot players.
 *
 * Portable: Can be adapted to other Teeworlds-based servers.
 */
#ifndef GAME_SERVER_RL_SERVER_SHM_H
#define GAME_SERVER_RL_SERVER_SHM_H

#include <game/rl/rl_protocol.h>
#include <cstddef>

namespace rl {

class CServerShm
{
public:
	CServerShm();
	~CServerShm();

	// Initialize shared memory for N bots
	bool Init(int NumBots);
	void Shutdown();

	// Check if initialized
	bool IsInitialized() const { return m_Initialized; }

	// Get number of bots
	int NumBots() const { return m_NumBots; }

	// Access bot data
	SBotData *GetBotData(int BotIndex);
	const SBotData *GetBotData(int BotIndex) const;

	// Access header
	SServerRLHeader *GetHeader();
	const SServerRLHeader *GetHeader() const;

	// Synchronization
	void SignalObservationsReady();  // Server has written observations
	bool AreActionsReady();          // Check if Python has written actions
	void ClearActionsReady();        // Clear for next tick
	bool IsShutdownRequested();      // Check if Python requested shutdown

	// Wait for actions with timeout (returns false on timeout)
	bool WaitForActions(int TimeoutMs);

private:
	char m_aShmPath[256];
	int m_ShmFd;
	void *m_pShmPtr;
	size_t m_ShmSize;
	int m_NumBots;
	bool m_Initialized;
};

} // namespace rl

#endif // GAME_SERVER_RL_SERVER_SHM_H
