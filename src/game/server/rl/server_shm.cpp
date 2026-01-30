/* DDNet Server RL - Shared Memory Communication Implementation */
#include "server_shm.h"

#include <base/system.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <thread>

namespace rl {

CServerShm::CServerShm() :
	m_ShmFd(-1),
	m_pShmPtr(nullptr),
	m_ShmSize(0),
	m_NumBots(0),
	m_Initialized(false)
{
	m_aShmPath[0] = '\0';
}

CServerShm::~CServerShm()
{
	Shutdown();
}

bool CServerShm::Init(int NumBots)
{
	if(m_Initialized)
	{
		dbg_msg("rl/server", "Shared memory already initialized");
		return false;
	}

	if(NumBots <= 0 || NumBots > MAX_RL_BOTS)
	{
		dbg_msg("rl/server", "Invalid number of bots: %d (max %d)", NumBots, MAX_RL_BOTS);
		return false;
	}

	m_NumBots = NumBots;
	m_ShmSize = sizeof(SServerRLSharedMemory);

	// Use fixed name from protocol (shm_open expects name without /dev/shm/)
	str_copy(m_aShmPath, GetServerRLShmName(), sizeof(m_aShmPath));

	// Remove existing shared memory if it exists
	shm_unlink(m_aShmPath);

	// Create shared memory object
	m_ShmFd = shm_open(m_aShmPath, O_CREAT | O_RDWR, 0666);
	if(m_ShmFd == -1)
	{
		dbg_msg("rl/server", "Failed to create shared memory: %s", m_aShmPath);
		return false;
	}

	// Set size
	if(ftruncate(m_ShmFd, m_ShmSize) == -1)
	{
		dbg_msg("rl/server", "Failed to set shared memory size: %zu bytes", m_ShmSize);
		close(m_ShmFd);
		shm_unlink(m_aShmPath);
		m_ShmFd = -1;
		return false;
	}

	// Map shared memory
	m_pShmPtr = mmap(nullptr, m_ShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, m_ShmFd, 0);
	if(m_pShmPtr == MAP_FAILED)
	{
		dbg_msg("rl/server", "Failed to map shared memory");
		close(m_ShmFd);
		shm_unlink(m_aShmPath);
		m_pShmPtr = nullptr;
		m_ShmFd = -1;
		return false;
	}

	// Initialize to zero
	mem_zero(m_pShmPtr, m_ShmSize);

	// Set header
	SServerRLHeader *pHeader = GetHeader();
	pHeader->m_NumBots = NumBots;
	pHeader->m_ServerReady = 0;
	pHeader->m_PythonReady = 0;
	pHeader->m_Shutdown = 0;

	m_Initialized = true;

	dbg_msg("rl/server", "Shared memory initialized: %s", m_aShmPath);
	dbg_msg("rl/server", "  Bots: %d, Size: %zu bytes (~%.1f KB)",
		NumBots, m_ShmSize, m_ShmSize / 1024.0f);

	return true;
}

void CServerShm::Shutdown()
{
	if(!m_Initialized)
		return;

	// Signal shutdown to Python
	if(m_pShmPtr)
	{
		SServerRLHeader *pHeader = GetHeader();
		pHeader->m_Shutdown = 1;
		pHeader->m_ServerReady = 1; // Wake up Python if waiting
		__sync_synchronize();
	}

	// Unmap and close
	if(m_pShmPtr && m_pShmPtr != MAP_FAILED)
	{
		munmap(m_pShmPtr, m_ShmSize);
		m_pShmPtr = nullptr;
	}

	if(m_ShmFd != -1)
	{
		close(m_ShmFd);
		m_ShmFd = -1;
	}

	// Remove shared memory
	if(m_aShmPath[0] != '\0')
	{
		shm_unlink(m_aShmPath);
	}

	m_Initialized = false;
	dbg_msg("rl/server", "Shared memory shutdown complete");
}

SServerRLHeader *CServerShm::GetHeader()
{
	if(!m_pShmPtr)
		return nullptr;
	return &static_cast<SServerRLSharedMemory *>(m_pShmPtr)->m_Header;
}

const SServerRLHeader *CServerShm::GetHeader() const
{
	if(!m_pShmPtr)
		return nullptr;
	return &static_cast<const SServerRLSharedMemory *>(m_pShmPtr)->m_Header;
}

SBotData *CServerShm::GetBotData(int BotIndex)
{
	if(!m_pShmPtr || BotIndex < 0 || BotIndex >= m_NumBots)
		return nullptr;
	return &static_cast<SServerRLSharedMemory *>(m_pShmPtr)->m_aBots[BotIndex];
}

const SBotData *CServerShm::GetBotData(int BotIndex) const
{
	if(!m_pShmPtr || BotIndex < 0 || BotIndex >= m_NumBots)
		return nullptr;
	return &static_cast<const SServerRLSharedMemory *>(m_pShmPtr)->m_aBots[BotIndex];
}

void CServerShm::SignalObservationsReady()
{
	if(!m_pShmPtr)
		return;

	SServerRLHeader *pHeader = GetHeader();

	// Memory barrier to ensure all observation data is written
	__sync_synchronize();

	pHeader->m_ServerReady = 1;
}

bool CServerShm::AreActionsReady()
{
	if(!m_pShmPtr)
		return false;

	const SServerRLHeader *pHeader = GetHeader();
	return pHeader->m_PythonReady != 0;
}

void CServerShm::ClearActionsReady()
{
	if(!m_pShmPtr)
		return;

	SServerRLHeader *pHeader = GetHeader();
	pHeader->m_PythonReady = 0;
	pHeader->m_ServerReady = 0;
}

bool CServerShm::IsShutdownRequested()
{
	if(!m_pShmPtr)
		return false;

	const SServerRLHeader *pHeader = GetHeader();
	return pHeader->m_Shutdown != 0;
}

bool CServerShm::WaitForActions(int TimeoutMs)
{
	if(!m_pShmPtr)
		return false;

	auto start = std::chrono::steady_clock::now();
	auto timeout = std::chrono::milliseconds(TimeoutMs);

	while(!AreActionsReady())
	{
		if(IsShutdownRequested())
			return false;

		auto elapsed = std::chrono::steady_clock::now() - start;
		if(elapsed >= timeout)
			return false;

		// Short sleep to avoid busy-waiting
		std::this_thread::sleep_for(std::chrono::microseconds(100));
	}

	return true;
}

} // namespace rl
